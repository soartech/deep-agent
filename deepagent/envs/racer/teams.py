from deepagent.envs.racer.units import RacerUgv, Unit, TeamNumber
from typing import Union, Type


class Team:
    # def __init__(self, ally_unit_names, enemy_unit_names, team_number: TeamNumber, agent_controlled: bool):
    def __init__(self, ally_unit_names, team_number: TeamNumber, agent_controlled: bool):
        self.team_number = team_number
        self.unit_classes = [Team.get_unit_class(name) for name in ally_unit_names]
        self.ally_unit_type_strings = [self.get_unit_type_string(cls.__name__) for cls in self.unit_classes]
        # self.enemy_agent_names = enemy_unit_names
        self._units = {}
        self.enemies_seen = set()
        self._obs_value_names = None
        self.agent_controlled = agent_controlled

    @staticmethod
    def get_unit_class(class_name: str) -> Type[Unit]:
        if class_name == 'racerugv':
            return RacerUgv
        raise ValueError(f"Don't know what class type to return for {class_name}")

    def clear(self):
        for unit in self._units.values():
            unit.remove()
        self._units = {}
        self.enemies_seen = set()

    def add_unit(self, unit):
        self._units[unit.tag] = unit

    @property
    def units(self):
        return list(self._units.values())

    @property
    def visible_units(self):
        return [u for u in self._units.values() if u.visible]

    def has_unit(self, unit_tag):
        return unit_tag in self._units

    def get_unit(self, unit_tag):
        if unit_tag in self._units:
            return self._units[unit_tag]
        return None

    def remove_dead_units(self):
        units_to_remove = set()
        for unit_tag, unit in self._units.items():
            if unit.is_dead:
                units_to_remove.add(unit_tag)
        for unit_tag in units_to_remove:
            self._units.pop(unit_tag)

    def get_unit_type_string(self, unit_type: str):
        return f'{unit_type.lower()}_team_{self.team_id}'

    @property
    def team_id(self) -> int:
        return self.team_number.value

    @property
    def enemy_id(self) -> int:
        return self.team_number.enemy().value

    def remove_unit(self, unit_tag):
        unit = self._units.pop(unit_tag)
        unit.remove()

class Teams:
    def __init__(self, team1: Team, team2: Team):
        self.team1 = team1
        self.team2 = team2

        if self.team1 is None or self.team2 is None:
            raise ValueError(f'Both teams must be set, but team1={self.team1}, team2={self.team2}')
        if self.team1.team_number != TeamNumber.ONE or self.team2.team_number != TeamNumber.TWO:
            raise ValueError(
                f'Teams have incorrect position values: team1={self.team1.team_number} team2={self.team2.team_number}')

    def get_team(self, player_id: Union[int, TeamNumber]):
        if type(player_id) == TeamNumber:
            player_id = player_id.value
        if player_id == 1:
            return self.team1
        elif player_id == 2:
            return self.team2

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= 2:
            raise StopIteration
        team = self.team1 if self._idx == 0 else self.team2
        self._idx += 1
        return team

    def get_unit(self, unit_tag):
        unit = self.team1.get_unit(unit_tag)
        if unit is None:
            unit = self.team2.get_unit(unit_tag)
        return unit

    def remove_dead_units(self):
        for team in [self.team1, self.team2]:
            team.remove_dead_units()

    def clear(self):
        for team in [self.team1, self.team2]:
            team.clear()

    def agent_controlled(self):
        return [t for t in self if t.agent_controlled]