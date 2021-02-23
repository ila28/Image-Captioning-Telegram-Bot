import os
import pickle
from io import BytesIO
import sys

import asyncio
from aiogram import Bot, Dispatcher, executor, filters, types
import numpy as np
from PIL import Image
import tensorflow as tf
from google_trans_new import google_translator
from gtts import gTTS
import requests

from model import load_image, evaluate


API_TOKEN = 'Токет Бота'
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
translator = google_translator()
__text = 'еще разок? 😏'
___text = '🙄🙄🙄...'


if not os.path.exists('audio'):
    os.mkdir('audio')

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    _user_name = message.from_user.first_name
    _text = 'привет, %s ! меня зовут бот Аннотолий🤓 ' %_user_name
    await message.reply(_text)

def translate(word):
    result = translator.translate(word, lang_tgt='ru')
    return result[:-1]

   

@dp.message_handler(content_types=['photo'])
async def echo(message):

    # media = types.MediaGroup()
    
    WIDTH = message.photo[-1].width
    HEIGHT = message.photo[-1].height

    byteImgIO = BytesIO()
    await message.photo[-1].download(byteImgIO)
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()

    dataBytesIO =  BytesIO(byteImg)

    

    # im = Image.frombuffer('RGB', (WIDTH, HEIGHT), dataBytesIO.getvalue())
    im = np.array(Image.open(dataBytesIO))
    res_l = evaluate(im)
    res_l = [x for x in res_l if x not in ['<unk>', '<end>']]
    result = ' '.join(res_l)
    result_ru = translate(result)
    print('result: ', result, result_ru)

    try:
        r = requests.post(
            'https://ttsmp3.com/makemp3_new.php',
            data={'msg': result_ru, 'lang': 'Maxim', 'source': 'ttsmp3'},
            headers={'User-agent': 'Mozilla/5.0'},
            config={'verbose': sys.stderr}
        )
        print(r)
        print(r.json())
        r1 = requests.get(r.json()['URL'])
        with open('./audio/audio.ogg', 'wb') as f:
            f.write(r1.content)
    except:
        var = gTTS(text = result_ru,lang = 'ru') 
        var.save('./audio/audio.ogg')


    await message.answer_voice(types.InputFile(
        './audio/audio.ogg'))
    await message.answer("''" + result_ru + "''") 
    await message.answer(__text)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)