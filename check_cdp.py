# -*- coding: utf-8 -*-
# ЦИАН через реальный Chrome (CDP) с лимитом ≤ 60 сек на ссылку
# Детектор снятия: фраза "Объявление снято с публикации"
# Без длинных ожиданий: короткие safe-ожидания, без лишних TimeoutError

import time, random
from pathlib import Path
from datetime import datetime

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# === Пути ===
INPUT_XLSX = r"C:\Users\Я\Documents\Учеба\Магистерская диссертация ВШЭ\Актуализация объявлений (Python)\Тест объявлений.xlsx"
OUTPUT_XLSX = r"C:\Users\Я\Documents\Учеба\Магистерская диссертация ВШЭ\Актуализация объявлений (Python)\Тест объявлений_результат.xlsx"
CDP_URL = "http://localhost:9222"  # Chrome запущен с --remote-debugging-port=9222

# === Настройки времени ===
TOTAL_PER_LINK_SEC = 60  # общий лимит на ссылку (жёстко ≤ 60 сек)
GOTO_TIMEOUT_MS = 45_000  # таймаут page.goto (входит в общий лимит)
SHORT_WAIT_MS = 1_200  # короткая дорисовка (1.2 c)
PHRASE_WAIT_MS = 1_500  # ожидание видимости фразы (1.5 c)

# === Поведение ===
PHRASE = "Объявление снято с публикации"
SLEEP_MIN, SLEEP_MAX = 1.2, 2.4  # пауза между ссылками
DUMPS = Path("./cian_dumps");
DUMPS.mkdir(exist_ok=True)


def now():
    return time.monotonic()


def remaining_ms(start_t):
    # сколько времени осталось из per-link лимита
    rem = TOTAL_PER_LINK_SEC - (now() - start_t)
    return max(0, int(rem * 1000))


def is_waf(page) -> bool:
    html = page.content().lower().replace("\xa0", " ")
    return ("cian_waf_block" in html) or ("кажется, у вас включён vpn" in html)


def humanize_quick(page):
    # лёгкая «очеловечка» без потери времени
    try:
        page.mouse.move(200, 150)
        page.mouse.move(600, 300)
        page.mouse.wheel(0, random.randint(400, 900))
    except Exception:
        pass


def detect_removed(page, start_t) -> bool:
    """Безопасная проверка фразы: короткие ожидания, никаких длинных блокировок."""
    # даём странице дорисоваться коротко (вписываемся в дедлайн)
    rem = remaining_ms(start_t)
    if rem <= 0:
        return False
    page.wait_for_timeout(min(SHORT_WAIT_MS, rem))

    # 1) видимый текст (короткий таймаут)
    try:
        rem = remaining_ms(start_t)
        if rem > 0:
            page.get_by_text(PHRASE, exact=False).wait_for(state="visible", timeout=min(PHRASE_WAIT_MS, rem))
            return True
    except PWTimeout:
        pass
    except Exception:
        pass

    # 2) фоллбек — по HTML (моментально)
    html = page.content().lower().replace("\xa0", " ")
    return PHRASE.lower() in html


def dump(page, url, tag):
    sid = url.rstrip("/").split("/")[-1] or "noid"
    (DUMPS / f"{sid}_{tag}.html").write_text(page.content(), encoding="utf-8")
    page.screenshot(path=str(DUMPS / f"{sid}_{tag}.png"))


def main():
    df = pd.read_excel(INPUT_XLSX)
    if "link" not in df.columns:
        raise ValueError("В Excel нужен столбец 'link'.")

    today = datetime.today().strftime("%Y-%m-%d")
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_URL)
        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
        page = ctx.new_page()
        # короткий глобальный таймаут для любых операций
        page.set_default_timeout(15_000)

        for i, raw_url in enumerate(df["link"], start=1):
            url = str(raw_url).strip()
            if not url:
                rows.append({"link": url, "Статус": "Ошибка: пустая ссылка", "Дата проверки": today})
                continue

            print(f"[{i}/{len(df)}] {url}")
            start_t = now()

            # быстрый сброс вкладки
            try:
                page.goto("about:blank", timeout=10_000)
            except Exception:
                pass

            # навигация с ограничением по дедлайну
            try:
                rem = remaining_ms(start_t)
                if rem <= 0:
                    rows.append({"link": url, "Статус": "Не проверено (лимит 60с)", "Дата проверки": today})
                    continue
                page.goto(url, timeout=min(GOTO_TIMEOUT_MS, rem), wait_until="domcontentloaded")
            except PWTimeout:
                rows.append({"link": url, "Статус": "Не проверено (таймаут загрузки)", "Дата проверки": today})
                continue
            except Exception as e:
                rows.append({"link": url, "Статус": f"Ошибка: {type(e).__name__}", "Дата проверки": today})
                continue

            # быстрая «очеловечка»
            humanize_quick(page)

            # WAF?
            if is_waf(page):
                rows.append({"link": url, "Статус": "Не проверено (WAF/VPN)", "Дата проверки": today})
                dump(page, url, "waf")
                time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
                continue

            # детект снятия — всё в пределах оставшегося времени
            removed = detect_removed(page, start_t)
            status = "Снято" if removed else "Активно"
            rows.append({"link": url, "Статус": status, "Дата проверки": today})

            if status == "Активно":
                dump(page, url, "active")

            # пауза между ссылками (не выходим за лимит, это уже вне проверки)
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        browser.close()

    out = df.merge(pd.DataFrame(rows), on="link", how="left")
    out.to_excel(OUTPUT_XLSX, index=False)
    print(f"✅ Готово! Результат сохранён в: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
