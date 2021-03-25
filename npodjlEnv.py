from julia import npod
from julia import Main


class NpodJLEnv:
    def __init__(self):
        self.pkdata_file = "data/data_1comp_neely.csv"
        self.sim_file = ""
        self.a = [0.001, 125]
        self.b = [2, 625]
        self.c0 = 0
        self.c1 = 0.1
        self.reward = 0
        # Meta parameters
        self.inc_rate = 1.0/10.0
        self.dec_rate = 1.0/11.0

    def encoded_state(self):
        return("%f_%f_%f_%f_%f_%f" % (self.a[0], self.a[1], self.b[0], self.b[1], self.c0, self.c1))

    def reset(self):
        self.a = [0.001, 125]
        self.b = [2, 625]
        self.c0 = 0
        self.c1 = 0.1
        self.reward = 0

    def run(self, action_i):
        action = self.actions()[action_i]
        if action == "inc_a1":
            self.a[0] = self.a[0] + self.inc_rate * self.a[0]
        elif action == "inc_b1":
            self.b[0] = self.b[0] + self.inc_rate * self.b[0]
        elif action == "inc_a2":
            self.a[1] = self.a[1] + self.inc_rate * self.a[1]
        elif action == "inc_b2":
            self.b[1] = self.b[1] + self.inc_rate * self.b[1]
        # elif action == "inc_c0":
        #     self.c0 = self.c0 + self.inc_rate * self.c0
        # elif action == "inc_c1":
        #     self.c1 = self.c1 + self.inc_rate * self.c1
        elif action == "dec_a1":
            self.a[0] = self.a[0] - self.dec_rate * self.a[0]
        elif action == "dec_b1":
            self.b[0] = self.b[0] - self.dec_rate * self.b[0]
        elif action == "dec_a2":
            self.a[1] = self.a[1] - self.dec_rate * self.a[1]
        elif action == "dec_b2":
            self.b[1] = self.b[1] - self.dec_rate * self.b[1]
        # elif action == "dec_c0":
        #     self.c0 = self.c0 - self.dec_rate * self.c0
        # elif action == "dec_c1":
        #     self.c1 = self.c1 - self.dec_rate * self.c1

        (cycles, theta, w, fobj, _) = npod.run(self.pkdata_file,
                                               self.a, self.b, self.c0, self.c1, 0, 51)
        Main.fobj = fobj
        fobj_val = Main.eval("fobj[]")
        return_dic = {'cycles': cycles, 'theta': theta, 'w': w, 'fobj': fobj_val, 'a': self.a, 'b': self.b, 'c0': self.c0,
                      'c1': self.c1, 'reward': fobj_val - self.reward}
        # if self.reward < result_dic["LogLikelihood"]:
        self.reward = fobj_val
        return(return_dic)

    def ver(self):
        return(0.8)

    def actions(self):
        return([
            "dec_a2",
            "dec_b2",
            "inc_a1",
            "inc_b1",
            "dec_a1",
            "dec_b1",
            "inc_a2",
            "inc_b2"
            # "inc_c0",
            # "inc_c1",
            # "dec_c0",
            # "dec_c1"

        ])
