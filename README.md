# Comment-Controversy-Detector
A project for detecting the controversy associated with scraped youtube comments

`data_annotating.py` reads CSV-style rows (`id,text,timestamp`). Rows may span multiple physical lines; it buffers until a line ends with `,YYYY-MM-DDTHH:MM:SSZ`, then replaces newlines in the comment with spaces. It writes one cleaned comment per logical row.

It strips the id and timestamp fields, removes commas inside the comment text, replaces emojis with `[e]` in the saved text, and uses Lingua so only longer lines must test as English (very short lines skip detection). It then drops non-ASCII characters.

Install dependencies with `pip install -r requirements.txt`, then run `python3 data_annotating.py input.csv output.txt`.
