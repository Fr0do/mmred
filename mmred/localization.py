"""Localization support for MMReD benchmark.

Provides Language configurations for generating questions and sequences
in different languages, with proper grammatical handling.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Language:
    """Language configuration for MMReD generation.

    Attributes:
        name: Language code ("en" or "ru")
        chars: Character names in this language
        rooms: Room names in this language
        nobody: "Nobody" in this language
        char_genders: Mapping of character name to gender ('m' or 'f')
        room_prepositional: Mapping of room name to prepositional form ("in the X")
    """
    name: str
    chars: list[str]
    rooms: list[str]
    nobody: str
    char_genders: dict[str, str] = field(default_factory=dict)
    room_prepositional: dict[str, str] = field(default_factory=dict)

    @classmethod
    def english(cls) -> "Language":
        chars = ["Sandra", "Mary", "John", "Daniel", "Michael"]
        rooms = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
        return cls(
            name="en",
            chars=chars,
            rooms=rooms,
            nobody="Nobody",
            char_genders={c: "f" if c in ("Sandra", "Mary") else "m" for c in chars},
            room_prepositional={r: r for r in rooms},
        )

    @classmethod
    def russian(cls) -> "Language":
        chars = ["Сандра", "Мария", "Иван", "Даниил", "Михаил"]
        rooms = ["Кухня", "Ванная", "Сад", "Офис", "Спальня", "Коридор"]
        return cls(
            name="ru",
            chars=chars,
            rooms=rooms,
            nobody="Никто",
            char_genders={
                "Сандра": "f",
                "Мария": "f",
                "Иван": "m",
                "Даниил": "m",
                "Михаил": "m",
            },
            room_prepositional={
                "Кухня": "на Кухне",
                "Ванная": "в Ванной",
                "Сад": "в Саду",
                "Офис": "в Офисе",
                "Спальня": "в Спальне",
                "Коридор": "в Коридоре",
            },
        )

    @classmethod
    def from_code(cls, code: str) -> "Language":
        if code == "ru":
            return cls.russian()
        return cls.english()

    def _gender(self, char: str) -> str:
        return self.char_genders.get(char, "m")

    def _verb(self, char: str, m_form: str, f_form: str) -> str:
        return f_form if self._gender(char) == "f" else m_form

    def _prep(self, room: str) -> str:
        return self.room_prepositional.get(room, f"в {room}")

    def translate_question(self, question: str) -> str:
        """Translate an English-template question (with Russian entity names) to Russian.

        The question is generated with Russian char/room names embedded in English templates.
        This method replaces the English template parts with proper Russian grammar.
        """
        if self.name == "en":
            return question

        chars_pattern = "|".join(re.escape(c) for c in self.chars)
        rooms_pattern = "|".join(re.escape(r) for r in self.rooms)

        # Helper to extract entities from a regex match
        def _prep_room(room_name):
            return self._prep(room_name)

        # --- NIAH questions ---

        # "In which room did {char} first appear?"
        m = re.fullmatch(rf"In which room did ({chars_pattern}) first appear\?", question)
        if m:
            c = m.group(1)
            v = self._verb(c, "появился", "появилась")
            return f"В какой комнате {c} {v} впервые?"

        # "In which room was {char} at the final step?"
        m = re.fullmatch(rf"In which room was ({chars_pattern}) at the final step\?", question)
        if m:
            c = m.group(1)
            v = self._verb(c, "был", "была")
            return f"В какой комнате {v} {c} на последнем шаге?"

        # "In which room was {c0} when {c1} first appeared in the {room}?"
        m = re.fullmatch(
            rf"In which room was ({chars_pattern}) when ({chars_pattern}) first appeared in the ({rooms_pattern})\?",
            question,
        )
        if m:
            c0, c1, room = m.group(1), m.group(2), m.group(3)
            v0 = self._verb(c0, "был", "была")
            v1 = self._verb(c1, "появился", "появилась")
            return f"В какой комнате {v0} {c0}, когда {c1} впервые {v1} {_prep_room(room)}?"

        # "In which room was {c0} when {c1} made their final appearance in the {room}?"
        m = re.fullmatch(
            rf"In which room was ({chars_pattern}) when ({chars_pattern}) made their final appearance in the ({rooms_pattern})\?",
            question,
        )
        if m:
            c0, c1, room = m.group(1), m.group(2), m.group(3)
            v0 = self._verb(c0, "был", "была")
            v1 = self._verb(c1, "появился", "появилась")
            return f"В какой комнате {v0} {c0}, когда {c1} в последний раз {v1} {_prep_room(room)}?"

        # "In which room was {char} at step {N}?"
        m = re.fullmatch(rf"In which room was ({chars_pattern}) at step (\d+)\?", question)
        if m:
            c, n = m.group(1), m.group(2)
            v = self._verb(c, "был", "была")
            return f"В какой комнате {v} {c} на шаге {n}?"

        # "Who was the first to appear in the {room}?"
        m = re.fullmatch(rf"Who was the first to appear in the ({rooms_pattern})\?", question)
        if m:
            room = m.group(1)
            return f"Кто первым появился {_prep_room(room)}?"

        # "Who was the last to appear in the {room}?"
        m = re.fullmatch(rf"Who was the last to appear in the ({rooms_pattern})\?", question)
        if m:
            room = m.group(1)
            return f"Кто последним появился {_prep_room(room)}?"

        # "Who was in the {room_0} when {char} first appeared in the {room_1}?"
        m = re.fullmatch(
            rf"Who was in the ({rooms_pattern}) when ({chars_pattern}) first appeared in the ({rooms_pattern})\?",
            question,
        )
        if m:
            r0, c, r1 = m.group(1), m.group(2), m.group(3)
            v = self._verb(c, "появился", "появилась")
            return f"Кто был {_prep_room(r0)}, когда {c} впервые {v} {_prep_room(r1)}?"

        # "Who was in the {room_0} when {char} made their final appearance in the {room_1}?"
        m = re.fullmatch(
            rf"Who was in the ({rooms_pattern}) when ({chars_pattern}) made their final appearance in the ({rooms_pattern})\?",
            question,
        )
        if m:
            r0, c, r1 = m.group(1), m.group(2), m.group(3)
            v = self._verb(c, "появился", "появилась")
            return f"Кто был {_prep_room(r0)}, когда {c} в последний раз {v} {_prep_room(r1)}?"

        # "Who was in the {room} at step {N}?"
        m = re.fullmatch(rf"Who was in the ({rooms_pattern}) at step (\d+)\?", question)
        if m:
            room, n = m.group(1), m.group(2)
            return f"Кто был {_prep_room(room)} на шаге {n}?"

        # "Who was in the same room as {char} at step {N}?"
        m = re.fullmatch(rf"Who was in the same room as ({chars_pattern}) at step (\d+)\?", question)
        if m:
            c, n = m.group(1), m.group(2)
            return f"Кто был в одной комнате с {c} на шаге {n}?"

        # "How many characters were in the {room_0} when {char} first appeared in the {room_1}?"
        m = re.fullmatch(
            rf"How many characters were in the ({rooms_pattern}) when ({chars_pattern}) first appeared in the ({rooms_pattern})\?",
            question,
        )
        if m:
            r0, c, r1 = m.group(1), m.group(2), m.group(3)
            v = self._verb(c, "появился", "появилась")
            return f"Сколько персонажей было {_prep_room(r0)}, когда {c} впервые {v} {_prep_room(r1)}?"

        # "How many characters were in the {room_0} when {char} made their final appearance in the {room_1}?"
        m = re.fullmatch(
            rf"How many characters were in the ({rooms_pattern}) when ({chars_pattern}) made their final appearance in the ({rooms_pattern})\?",
            question,
        )
        if m:
            r0, c, r1 = m.group(1), m.group(2), m.group(3)
            v = self._verb(c, "появился", "появилась")
            return f"Сколько персонажей было {_prep_room(r0)}, когда {c} в последний раз {v} {_prep_room(r1)}?"

        # "How many other characters were in the same room as {char} at step {N}?"
        m = re.fullmatch(
            rf"How many other characters were in the same room as ({chars_pattern}) at step (\d+)\?",
            question,
        )
        if m:
            c, n = m.group(1), m.group(2)
            return f"Сколько других персонажей было в одной комнате с {c} на шаге {n}?"

        # "How many rooms were empty at step {N}?"
        m = re.fullmatch(r"How many rooms were empty at step (\d+)\?", question)
        if m:
            n = m.group(1)
            return f"Сколько комнат были пустыми на шаге {n}?"

        # --- DC questions ---

        # Suffix handling: "?" or " between steps X and Y?"
        def _translate_suffix(suffix: str) -> str:
            m_s = re.fullmatch(r" between steps (\d+) and (\d+)\?", suffix)
            if m_s:
                return f" между шагами {m_s.group(1)} и {m_s.group(2)}?"
            return "?"

        # "Which room was empty for {more/fewer} steps than the other rooms{suffix}"
        m = re.fullmatch(
            rf"Which room was empty for (more|fewer) steps than the other rooms( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            comp = "больше" if m.group(1) == "more" else "меньше"
            suffix = _translate_suffix(m.group(2))
            return f"Какая комната была пустой {comp} шагов, чем остальные{suffix}"

        # "In which room did {char} spend the {most/least amount of} time{suffix}"
        m = re.fullmatch(
            rf"In which room did ({chars_pattern}) spend the (most|least amount of) time( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            c = m.group(1)
            comp = "больше всего" if m.group(2) == "most" else "меньше всего"
            suffix = _translate_suffix(m.group(3))
            return f"В какой комнате {c} провёл(а) {comp} времени{suffix}"

        # "Which room was crowded ({N} or more people in one room) for the most steps{suffix}"
        m = re.fullmatch(
            rf"Which room was crowded \((\d+) or more people in one room\) for the most steps( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            n_crowd = m.group(1)
            suffix = _translate_suffix(m.group(2))
            return f"Какая комната была переполнена ({n_crowd} или более человек) наибольшее количество шагов{suffix}"

        # "Who spent the {most/least amount of} time alone in the {room}{suffix}"
        m = re.fullmatch(
            rf"Who spent the (most|least amount of) time alone in the ({rooms_pattern})( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            comp = "больше всего" if m.group(1) == "most" else "меньше всего"
            room = m.group(2)
            suffix = _translate_suffix(m.group(3))
            return f"Кто провёл {comp} времени в одиночестве {_prep_room(room)}{suffix}"

        # "Who spent the {most/least amount of} time alone in the rooms{suffix}"
        m = re.fullmatch(
            rf"Who spent the (most|least amount of) time alone in the rooms( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            comp = "больше всего" if m.group(1) == "most" else "меньше всего"
            suffix = _translate_suffix(m.group(2))
            return f"Кто провёл {comp} времени в одиночестве{suffix}"

        # "With whom did {char} spend the {most/least amount of} time together in the same room{suffix}"
        m = re.fullmatch(
            rf"With whom did ({chars_pattern}) spend the (most|least amount of) time together in the same room( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            c = m.group(1)
            comp = "больше всего" if m.group(2) == "most" else "меньше всего"
            suffix = _translate_suffix(m.group(3))
            return f"С кем {c} провёл(а) {comp} времени вместе в одной комнате{suffix}"

        # "How many steps did {char} spend in the {room}{suffix}"
        m = re.fullmatch(
            rf"How many steps did ({chars_pattern}) spend in the ({rooms_pattern})( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            c, room = m.group(1), m.group(2)
            suffix = _translate_suffix(m.group(3))
            return f"Сколько шагов {c} провёл(а) {_prep_room(room)}{suffix}"

        # "How many different rooms did {char} visit{suffix}"
        m = re.fullmatch(
            rf"How many different rooms did ({chars_pattern}) visit( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            c = m.group(1)
            suffix = _translate_suffix(m.group(2))
            return f"Сколько разных комнат посетил(а) {c}{suffix}"

        # "How many times did a crowd ({N} or more people in one room) appear{suffix}"
        m = re.fullmatch(
            rf"How many times did a crowd \((\d+) or more people in one room\) appear( between steps \d+ and \d+\?|\?)",
            question,
        )
        if m:
            n_crowd = m.group(1)
            suffix = _translate_suffix(m.group(2))
            return f"Сколько раз возникала толпа ({n_crowd} или более человек в одной комнате){suffix}"

        # Fallback: return original question if no pattern matched
        return question
