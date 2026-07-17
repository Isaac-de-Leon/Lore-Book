#!/usr/bin/env python3
# fetch_card_prices.py — CLI for downloading card market prices.
#
# Thin wrapper around lorebook.core.price_fetcher, which the GUI also calls
# automatically whenever card_prices_<Game>.json is missing or older than a
# day. Prices are TCGplayer-sourced USD values (display only — never written
# to the collection CSV); currency_rates.json holds the daily USD exchange
# rates used for non-USD display.
#
#   python scripts/fetch_card_prices.py --game Lorcana
#   python scripts/fetch_card_prices.py --game Lorcana --skip-rates

import argparse
import os
import sys

try:
    from lorebook.core.price_fetcher import (
        GAMES, download_card_prices, download_currency_rates,
    )
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lorebook.core.price_fetcher import (
        GAMES, download_card_prices, download_currency_rates,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download card market prices → card_prices_<Game>.json.")
    p.add_argument("--game", default="Lorcana", help="Game folder name (default: Lorcana).")
    p.add_argument("--url", help="Override the price API base URL.")
    p.add_argument("--out", help="Output file (default: card_prices_<Game>.json).")
    p.add_argument("--skip-rates", action="store_true",
                   help="Don't refresh currency_rates.json afterwards.")
    args = p.parse_args(argv)

    if args.game.strip().lower() not in GAMES:
        print(f"No price source for {args.game!r} (supported: {', '.join(sorted(GAMES))}).",
              file=sys.stderr)
        return 1

    try:
        count = download_card_prices(
            args.game.strip().capitalize(),
            out_path=args.out,
            url=args.url,
            progress_callback=print,
        )
    except Exception as e:
        print(f"Price download failed: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {count} price entries.")

    if not args.skip_rates:
        try:
            download_currency_rates()
        except Exception as e:
            print(f"Currency-rate refresh failed (USD display still works): {e}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
