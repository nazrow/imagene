import os
import random
import asyncio
import traceback

from telegram.ext import ApplicationBuilder

from secretics import posting_dir, neurowisor_token
from prompts import clean_filename


async def post(context):
    while True:
        files = [f for f in os.listdir(posting_dir) if f.endswith('.jpg') and not 'POSTED' in f]
        print(f'{len(files)} unposted pics left...')
        if len(files) == 0:
            break
        file = random.choice(files)
        prompt = clean_filename(file).replace("-", "\-").replace(".", "\.")
        try:
            await context.bot.send_photo(
                chat_id=-1001892169588,
                photo=f'{posting_dir}/{file}',
                caption=prompt,
                parse_mode='MarkdownV2'
            )
            _file = file if len(file) < 100 else f'{file[:90]}... — {random.randint(1, 9999)}.jpg'
            try:
                os.remove(f'{posting_dir}/{_file.replace(".jpg", " POSTED.jpg")}')
            except:
                pass
            os.rename(f'{posting_dir}/{file}', f'{posting_dir}/{_file.replace(".jpg", " POSTED.jpg")}')
            wait = random.randint(1000, 10000)
            print(f'Waiting for {wait} secs...')
            await asyncio.sleep(wait)
        except:
            print(prompt)
            traceback.print_exc()
            wait = random.randint(1000, 10000)
            await asyncio.sleep(wait)


async def start_posting(context):
    asyncio.create_task(post(context))


app = ApplicationBuilder().token(neurowisor_token).build()
app.post_init = start_posting
app.run_polling()
