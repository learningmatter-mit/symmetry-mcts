from environments.environment import Y6Environment, PatentEnvironment


def y6_builder(reward_tp, output_dir, reduction):
    return Y6Environment(reward_tp, output_dir, reduction)


def patent_builder(reward_tp, output_dir, reduction):
    return PatentEnvironment(reward_tp, output_dir, reduction)
