import logging
from uuid import uuid4

from sqlalchemy import create_engine, insert, select
from sqlalchemy.orm import Session

from config import DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME
from models.models import feedback, model, user

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


def insert_data(stmt) -> None:
    with engine.connect() as conn:
        logging.warning(f"in connect {conn}")
        _ = conn.execute(stmt)

        conn.commit()


def get_or_create(telegram_user_id) -> None:
    with Session(engine) as session:
        check_user = session.query(user).filter_by(telegram_user_id=telegram_user_id).first()
        if check_user:
            logging.info("user exists")
        else:
            instance = insert(user).values(telegram_user_id=telegram_user_id)
            insert_data(instance)
            logging.info(f"new user created with id {telegram_user_id}")


def save_user_feedback(telegram_user_id: int, user_data: dict) -> None:
    get_or_create(telegram_user_id)

    with Session(engine) as session:
        user_id = session.execute(select(user.c.id).where(user.c.telegram_user_id == telegram_user_id)).first()

    with Session(engine) as session:
        model_id = session.execute(select(model.c.id).where(model.c.type == user_data['chosen_model'].upper())).first()

    feedback_stmt = insert(feedback).values(
        user_id=user_id[0],
        model_id=model_id[0],
        estimation=int(user_data['user_estimation']),
        length=int(user_data['length']),
        prefix=str((user_data['prefix_list'])),
        uuid=user_data['uuid'],
    )

    insert_data(feedback_stmt)


def save_user_feedback_with_model(telegram_user_id: int, user_data: dict, dl_model: str, model_uuid: uuid4, user_estimation) -> None:
    get_or_create(telegram_user_id)

    with Session(engine) as session:
        user_id = session.execute(select(user.c.id).where(user.c.telegram_user_id == telegram_user_id)).first()

    with Session(engine) as session:
        model_id = session.execute(select(model.c.id).where(model.c.type == dl_model.upper())).first()

    feedback_stmt = insert(feedback).values(
        user_id=user_id[0],
        model_id=model_id[0],
        estimation=user_estimation,
        length=int(user_data['length']),
        prefix=str((user_data['prefix_list'])),
        uuid=model_uuid,
    )

    insert_data(feedback_stmt)
