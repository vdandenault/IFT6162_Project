class NestedResponseSearch:
    def __init__(self, response, rollout, rank):
        self.response, self.rollout, self.rank = response, rollout, rank

    def plan(self, schedules, *, fixed_response=None):
        plans = []
        for schedule in schedules:
            response = self.response(schedule) if fixed_response is None else fixed_response
            plans.append(self.rollout(schedule, response))
        if not plans: raise ValueError('At least one leader schedule required')
        return min(plans, key=self.rank), len(plans)
