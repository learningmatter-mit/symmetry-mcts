from environments.environment import Y6Environment

def y6_builder(reward_tp, output_dir, reduction):
    return Y6Environment(reward_tp, output_dir, reduction) 
