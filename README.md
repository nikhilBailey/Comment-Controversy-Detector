# Youtube-Comment-Bot-Detection
Evan Gattis, Nikhil Bailey
A project for detecting suspected bots within scraped YouTube comments.

`data_annotating.py` reads CSV-style rows (`id,text,timestamp`). Rows may span multiple physical lines; it buffers until a line ends with `,YYYY-MM-DDTHH:MM:SSZ`, then replaces newlines in the comment with spaces. It writes one cleaned comment per logical row.

It strips the id and timestamp fields, removes commas inside the comment text, replaces emojis with `[e]` in the saved text, and uses Lingua so only longer lines must test as English (very short lines skip detection). It then drops non-ASCII characters.

Install dependencies with `pip install -r requirements.txt`, then run `python3 data_annotating.py input.csv output.txt`. For rows that end with `,timestamp,0` or `,timestamp,1` (pre-annotated labels), use `python3 data_annotating.py input.csv output.txt -a`: row boundaries use that suffix, and each output line is `cleansed_text,0` or `cleansed_text,1`.

To train models, use scripts/model_training/training.py. To adjust which models are trained, modify scripts/model_training/Model.py. Once trained, models results can be visualized with Visualizer.py. Results from this will be put in the visualizations folder with a timestamp. 
