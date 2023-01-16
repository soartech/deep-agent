from gym.envs.registration import register

register(
    id='DefeatFastVsSlow-v0',
    entry_point='deepagent.envs:FVSEnv',
)

register(
    id='DefeatFastVsSlow-v1',
    entry_point='deepagent.envs:FVSRawEnv',
)

register(
    id='DefeatZerglingsAndBanelingsTerrain-v0',
    entry_point='deepagent.envs:DZBTerrainEnv',
)

register(
    id='DefeatZerglingsAndBanelingsTerrain-v1',
    entry_point='deepagent.envs:DZBTerrainRawEnv',
)

register(
    id='Wedge-v0',
    entry_point='deepagent.envs:WedgeEnv',
)

register(
    id='HGroups-v0',
    entry_point='deepagent.envs:HGroupsEnv',
)

register(
    id='HGroup-v0',
    entry_point='deepagent.envs:HGroupEnv',
)

register(
    id='FilledWedge-v0',
    entry_point='deepagent.envs:FilledWedgeEnv',
)

register(
    id='ExploitUnitWeaknesses-v0',
    entry_point='deepagent.envs:BVCEnv',
)

register(
    id='DefeatZealots-v0',
    entry_point='deepagent.envs:MVZEnv',
)