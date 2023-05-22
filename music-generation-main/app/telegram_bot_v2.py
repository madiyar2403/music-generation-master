import asyncio
import logging
import random

from uuid import uuid4

from aiogram import Dispatcher, types, Bot
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import BotCommand

from app.data_manipulation import save_user_feedback_with_model
from midi2audio import FluidSynth

from config import TELEGRAM_BOT_TOKEN
from main import generate
from util import GenerateRequest, BASE_DIR

logger = logging.getLogger(__name__)

config = TELEGRAM_BOT_TOKEN

bot = Bot(token=config)

# available_model_names = ["rnn", "gan", "vae", "cnn"]
available_model_names = ["rnn", "vae", "cnn"]

estimations = [str(i) for i in range(10, 0, -1)]


class MusicGeneration(StatesGroup):
    waiting_for_model_length = State()
    waiting_for_model_user_estimation = State()


async def generate_music_by_input(random_sample, model, length):
    generate_request = GenerateRequest(
        model=model,
        length=length,
        prefix=random_sample,
        is_mid=True,
    )

    logging.info(f"generate_request {generate_request}")

    generated_music = await generate(generate_request)

    uuid = uuid4()
    file_path_midi = f'{BASE_DIR}/midi_generated_music/{model}/{uuid}.mid'
    file_path_wav = f'{BASE_DIR}/wav_rendered_music/{model}/{uuid}.wav'

    generated_music.write(file_path_midi)
    logging.warning(f"file_path_midi - {file_path_midi}")

    FluidSynth('FluidR3_GM.sf2')
    fs = FluidSynth()
    fs.midi_to_audio(file_path_midi, file_path_wav)

    return uuid, file_path_wav


async def send_welcome(message: types.Message):
    await message.reply("Привет, это бот для генерации музыки с помощью ИИ, чтобы начать введите команду /generate 🎵")


async def model_start(message: types.Message, state: FSMContext):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    await message.answer("Введите длину песни в секундах (до 40):", reply_markup=keyboard)
    await state.set_state(MusicGeneration.waiting_for_model_length.state)


async def model_length(message: types.Message, state: FSMContext):
    incorrect_format_message = "Пожалуйста, введите длину песни в секундах (до 40). Длина мелодии будет примерно (+-5) секунд 😁"

    if not message.text.isnumeric():
        await message.answer(incorrect_format_message)
        return

    if int(message.text) < 0 or int(message.text) > 40:
        await message.answer(incorrect_format_message)
        return

    length = int(message.text)

    await state.update_data(length=int(length) * 50)
    await message.answer(f"Вы выбрали длину {length}", reply_markup=types.ReplyKeyboardRemove())

    logging.info(f"length - {length}")

    user_data = await state.get_data()

    await message.answer(f"Вы выбрали длину {user_data['length'] / 50}. Вам будут предложены 3 мелодии разных моделей, оцените их, пожалуйста. "
                         f"Ожидайте сгенерированную музыку... 🎵🎵🎵")

    random_sample = random.sample(range(1, 255), 50)

    await state.update_data(prefix_list=random_sample)

    try:
        rnn_uuid, rnn_file_path_wav = await generate_music_by_input(random_sample, "rnn", user_data['length'])
        await state.update_data(rnn_uuid=rnn_uuid)

        vae_uuid, vae_file_path_wav = await generate_music_by_input(random_sample, "vae", user_data['length'])
        await state.update_data(vae_uuid=vae_uuid)

        cnn_uuid, cnn_file_path_wav = await generate_music_by_input(random_sample, "cnn", user_data['length'])
        await state.update_data(cnn_uuid=cnn_uuid)
    except Exception as e:
        logging.error(f"generated_music exception {e}")
        await message.answer("Ой, что-то пошло не так, попробуйте ввести другие входные данные. 🤔")
        await state.finish()
        return

    await bot.send_audio(chat_id=message.from_user.id, audio=open(rnn_file_path_wav, "rb"), performer=f"{rnn_uuid}", title="RNN")
    await bot.send_audio(chat_id=message.from_user.id, audio=open(vae_file_path_wav, "rb"), performer=f"{vae_uuid}", title="VAE")
    await bot.send_audio(chat_id=message.from_user.id, audio=open(cnn_file_path_wav, "rb"), performer=f"{cnn_uuid}", title="CNN")

    await message.answer("Пожалуйста, оцените сгенерированную музыку по десятибальной шкале RNN - VAE - CNN соответственно. "
                         "(3 цифры через запятую, пример: 10, 7, 8) 💩")

    await state.set_state(MusicGeneration.waiting_for_model_user_estimation.state)


async def model_user_estimation(message: types.Message, state: FSMContext):
    estimation_list = message.text
    incorrect_format_message = "Пожалуйста, выберите корректный балл через запятую по десятибальной шкале RNN - VAE - CNN соответственно. " \
                               "Пример: 5, 3, 9 💩"

    res_list = list()
    count_list = estimation_list.split(",")

    logging.info(f"count_list - {count_list}")

    if len(count_list) != 3:
        await message.answer(incorrect_format_message)
        return
    try:
        for estimation in count_list:
            if estimation not in estimations:
                await message.answer(incorrect_format_message)
                return
            res_list.append(int(estimation))
    except TypeError:
        await message.answer(incorrect_format_message)
        return

    await state.update_data(user_estimation=res_list)
    await message.answer("Спасибо за оценку, вы можете сгенерировать еще музыку, нажав команду /generate еще раз. 😊")

    user_data = await state.get_data()
    logging.info(f"user_data - {user_data}")

    save_user_feedback_with_model((int(message.from_user.id)), user_data, "rnn", user_data["rnn_uuid"], user_data["user_estimation"][0])
    save_user_feedback_with_model((int(message.from_user.id)), user_data, "vae", user_data["vae_uuid"], user_data["user_estimation"][1])
    save_user_feedback_with_model((int(message.from_user.id)), user_data, "cnn", user_data["cnn_uuid"], user_data["user_estimation"][2])

    await state.finish()


def register_handlers_model(dp: Dispatcher):
    dp.register_message_handler(model_start, commands="generate", state="*")
    dp.register_message_handler(model_length, state=MusicGeneration.waiting_for_model_length)
    dp.register_message_handler(model_user_estimation, state=MusicGeneration.waiting_for_model_user_estimation)


async def cmd_start(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer(
        "Введите команду /generate и выберите длину",
        reply_markup=types.ReplyKeyboardRemove()
    )


async def cmd_cancel(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer("Действие отменено.", reply_markup=types.ReplyKeyboardRemove())


def register_handlers_common(dp: Dispatcher):
    dp.register_message_handler(send_welcome, commands="help", state="*")
    dp.register_message_handler(cmd_start, commands="start", state="*")
    dp.register_message_handler(cmd_cancel, commands="cancel", state="*")
    dp.register_message_handler(cmd_cancel, Text(equals="отмена", ignore_case=True), state="*")


async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="/help", description="Помощь"),
        BotCommand(command="/start", description="Старт"),
        BotCommand(command="/generate", description="Сгенерировать музыку"),
        BotCommand(command="/cancel", description="Отменить текущее действие")
    ]
    await bot.set_my_commands(commands)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger.error("Starting bot")

    dp = Dispatcher(bot, storage=MemoryStorage())

    register_handlers_common(dp)
    register_handlers_model(dp)

    await set_commands(bot)

    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
