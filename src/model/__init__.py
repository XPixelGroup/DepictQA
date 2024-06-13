from .agent import DeepSpeedAgent
from .depictqa import DepictQA


def build_agent(args, training):
    model = DepictQA(args, training)
    agent = DeepSpeedAgent(model, args)
    return agent
