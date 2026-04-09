class MethodWrapper:
    def build_cache(self, cfg):
        raise NotImplementedError

    def build_context(self, cache, protocol, split="val"):
        raise NotImplementedError

    def run_with_config(self, ctx, config):
        raise NotImplementedError

    def evaluate(self, ctx, run_output, protocol):
        raise NotImplementedError

    def get_slots(self):
        raise NotImplementedError

    def estimate_cost(self, config):
        return 0.0
