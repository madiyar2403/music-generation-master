import asyncio
import logging

from uuid import uuid4

from aiogram import Dispatcher, types, Bot
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import BotCommand

from app.data_manipulation import save_user_feedback
from midi2audio import FluidSynth

from config import TELEGRAM_BOT_TOKEN
from main import generate
from util import GenerateRequest, BASE_DIR

logger = logging.getLogger(__name__)

config = TELEGRAM_BOT_TOKEN

bot = Bot(token=config)

available_model_names = ["rnn", "gan", "vae", "cnn"]
estimations = [str(i) for i in range(10, 0, -1)]


class MusicGeneration(StatesGroup):
    waiting_for_model_name = State()
    waiting_for_model_length = State()
    waiting_for_model_prefix = State()
    waiting_for_model_user_estimation = State()


async def send_welcome(message: types.Message):
    await message.reply("Привет, это бот для генерации музыки с помощью ИИ, чтобы начать введите команду /generate")


async def model_start(message: types.Message, state: FSMContext):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    for name in available_model_names:
        keyboard.add(name)
    await message.answer("Выберите модель для генерации музыки:", reply_markup=keyboard)
    await state.set_state(MusicGeneration.waiting_for_model_name.state)


async def model_chosen(message: types.Message, state: FSMContext):
    if message.text not in available_model_names:
        await message.answer("Пожалуйста, выберите модель, используя клавиатуру ниже.")
        return
    if message.text == "gan":
        await message.answer("Данная модель еще в разработке, выберите другую модель ;)")
        return

    await state.update_data(chosen_model=message.text.upper())

    await state.set_state(MusicGeneration.waiting_for_model_length.state)
    await message.answer("Теперь введите длину (100-1000):", reply_markup=types.ReplyKeyboardRemove())


async def model_length_chosen(message: types.Message, state: FSMContext):
    if not message.text.isnumeric():
        await message.answer("Пожалуйста, введите длину от 100 до 1000")
        return
    length = int(message.text)
    if length > 1000 or length < 100:
        await message.answer("Пожалуйста, введите корректный диапазон от 100 до 1000")
        return
    user_data = await state.get_data()
    await state.update_data(length=int(length) + 500)
    await state.set_state(MusicGeneration.waiting_for_model_prefix.state)

    await message.answer(f"Вы выбрали длину {length} для модели {user_data['chosen_model']}.\n"
                         f"Выберите префикс - введите корректную последовательность чисел через запятую (Пример: 1,2,3) от 1 до 255:",
                         reply_markup=types.ReplyKeyboardRemove())


async def model_prefix_chosen(message: types.Message, state: FSMContext):
    logging.info(f"message.text - {message.text} length - {len(message.text.split(','))}")

    incorrect_input_message = "Пожалуйста, введите корректную последовательность чисел через запятую (Пример: 1,2,3) от 1 до 255:"

    try:
        if int(message.text) > 0 and 255 < int(message.text):
            await message.answer(incorrect_input_message)
            return
        prefix_list = [int(message.text)]
        await state.update_data(prefix_list=prefix_list)
    except ValueError:
        try:
            prefix_list = [int(i) for i in message.text.split(",") if 0 < int(i) < 255]
            if not prefix_list:
                await message.answer(incorrect_input_message)
                return

            await state.update_data(prefix_list=prefix_list)
        except ValueError:
            await message.answer(incorrect_input_message)
            return

    user_data = await state.get_data()

    await message.answer(f"Вы выбрали {user_data['prefix_list']} префикс для модели {user_data['chosen_model']} с длиной "
                         f"{user_data['length'] - 500}. Ожидайте сгенерированную музыку...")

    generate_request = GenerateRequest(
        model=user_data['chosen_model'].lower(),
        length=user_data['length'],
        prefix=user_data['prefix_list'],
        is_mid=True,
    )

    logging.info(f"generate_request {generate_request}")

    try:
        generated_music = await generate(generate_request)
    except Exception as e:
        logging.error(f"generated_music exception {e}")
        await message.answer("Ой, что-то пошло не так, попробуйте ввести другие входные данные.")
        await state.finish()
        return

    uuid = uuid4()
    file_path_midi = f'{BASE_DIR}/midi_generated_music/{uuid}.mid'
    file_path_wav = f'{BASE_DIR}/wav_rendered_music/{uuid}.wav'

    generated_music.write(file_path_midi)
    logging.warning(f"file_path_midi - {file_path_midi}")

    await state.update_data(uuid=uuid)

    FluidSynth('FluidR3_GM.sf2')
    fs = FluidSynth()
    fs.midi_to_audio(file_path_midi, file_path_wav)

    await bot.send_audio(chat_id=message.from_user.id, audio=open(file_path_wav, "rb"), performer=f"{user_data['chosen_model'].upper()}",
                         title=f"{uuid}")

    keyboard = types.ReplyKeyboardMarkup()

    for name in estimations:
        keyboard.add(name)

    await message.answer("Пожалуйста, оцените данную сгенерированную музыку по десятибальной шкале.", reply_markup=keyboard)
    await state.set_state(MusicGeneration.waiting_for_model_user_estimation.state)


async def model_user_estimation(message: types.Message, state: FSMContext):
    if message.text not in estimations:
        await message.answer("Пожалуйста, выберите корректный балл(1-10).")
        return

    await state.update_data(user_estimation=message.text)
    await message.answer("Спасибо за оценку, вы можете сгенерировать еще музыку и попробовать другие модели, нажав команду /generate еще раз.",
                         reply_markup=types.ReplyKeyboardRemove())
    user_data = await state.get_data()
    logging.info(f"user_data - {user_data}")

    save_user_feedback(int(message.from_user.id), user_data)

    await state.finish()


def register_handlers_model(dp: Dispatcher):
    dp.register_message_handler(model_start, commands="generate", state="*")
    dp.register_message_handler(model_chosen, state=MusicGeneration.waiting_for_model_name)
    dp.register_message_handler(model_length_chosen, state=MusicGeneration.waiting_for_model_length)
    dp.register_message_handler(model_prefix_chosen, state=MusicGeneration.waiting_for_model_prefix)
    dp.register_message_handler(model_user_estimation, state=MusicGeneration.waiting_for_model_user_estimation)


async def cmd_start(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer(
        "Введите команду /generate и выберите модель для генерации музыки",
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
