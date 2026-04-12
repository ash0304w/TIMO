class MethodWrapper:
    def build_cache(self, cfg):
        raise NotImplementedError

    def build_protocol_view(self, protocol, cache):
        raise NotImplementedError

    def get_slots(self):
        raise NotImplementedError

    def materialize(self, prefix_action, ctx):
        raise NotImplementedError

    def run_with_action(self, ctx, action):
        raise NotImplementedError

    def evaluate(self, ctx, run_output, protocol):
        raise NotImplementedError

    def estimate_cost(self, action):
        return 0.0
