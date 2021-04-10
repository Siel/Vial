from julia import npod
from julia import npag
from julia import Main


class NpodJLEnv:
    def __init__(self, engine="npod"):
        self.pkdata_file = "data/data_1comp_neely.csv"
        self.sim_file = ""
        self.a = [0.001, 125, 0.001, 0.001]
        self.b = [2, 625, 100, 100]
        self.model = 2
        self.c0 = 0
        self.c1 = 0.1
        self.reward = 0
        # Meta parameters
        self.engine = engine
        self.inc_rate = 1.0/10.0
        self.dec_rate = 1.0/11.0

    def encoded_state(self):
        if self.model == 1:
            return("1:%f_%f_%f_%f" % (self.a[0], self.a[1], self.b[0], self.b[1]))
        elif self.model == 2:
            return("2:%f_%f_%f_%f_%f_%f_%f_%f" % (self.a[0], self.a[1], self.a[2], self.a[3], self.b[0], self.b[1], self.b[2], self.b[3]))

    def reset(self):
        self.a = [0.001, 125, 0.001, 0.001]
        self.b = [2, 625, 100, 100]
        self.model = 2
        self.c0 = 0
        self.c1 = 0.1
        self.reward = 0

    def run(self, action_i):
        action = self.actions()[action_i]
        # TODO: generalize the actions
        if action == "inc_a1":
            self.a[0] = self.a[0] + self.inc_rate * self.a[0]
        elif action == "inc_b1":
            self.b[0] = self.b[0] + self.inc_rate * self.b[0]
        elif action == "inc_a2":
            self.a[1] = self.a[1] + self.inc_rate * self.a[1]
        elif action == "inc_b2":
            self.b[1] = self.b[1] + self.inc_rate * self.b[1]
        elif action == "chg_model":
            self.model = 1 if self.model == 2 else 1
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
        elif action == "inc_a3":
            self.a[2] = self.a[2] + self.inc_rate * self.a[2]
        elif action == "inc_b3":
            self.b[2] = self.b[2] + self.inc_rate * self.b[2]
        elif action == "inc_a4":
            self.a[3] = self.a[3] + self.inc_rate * self.a[3]
        elif action == "inc_b4":
            self.b[3] = self.b[3] + self.inc_rate * self.b[3]
        elif action == "dec_a3":
            self.a[2] = self.a[2] - self.dec_rate * self.a[2]
        elif action == "dec_b3":
            self.b[2] = self.b[2] - self.dec_rate * self.b[2]
        elif action == "dec_a4":
            self.a[3] = self.a[3] - self.dec_rate * self.a[3]
        elif action == "dec_b4":
            self.b[3] = self.b[3] - self.dec_rate * self.b[3]

        a = self.a if self.model == 2 else self.a[:2]
        b = self.b if self.model == 2 else self.b[:2]
        if self.engine == "npod":
            (cycles, theta, w, fobj, _) = npod.run(self.model,
                                                   self.pkdata_file, a, b, self.c0, self.c1, 0, 200)
        elif self.engine == "npag":
            (cycles, theta, w, fobj, _) = npag.run(self.model,
                                                   self.pkdata_file, a, b, self.c0, self.c1, 0, 200)
        Main.fobj = fobj
        fobj_val = Main.eval("fobj[]")
        return_dic = {'cycles': cycles, 'theta': theta, 'w': w, 'fobj': fobj_val, 'a': self.a, 'b': self.b, 'c0': self.c0,
                      'c1': self.c1, 'reward': fobj_val - self.reward}
        # if self.reward < result_dic["LogLikelihood"]:
        self.reward = fobj_val
        return(return_dic)

    def ver(self):
        return(0.18)

    def actions(self):
        if self.model == 1:
            return([
                "dec_a2",
                "dec_b2",
                "inc_a1",
                "inc_b1",
                "dec_a1",
                "dec_b1",
                "inc_a2",
                "inc_b2",
                "chg_model"
                # "inc_c0",
                # "inc_c1",
                # "dec_c0",
                # "dec_c1"

            ])
        elif self.model == 2:
            return([
                "dec_a2",
                "dec_b2",
                "inc_a1",
                "inc_b1",
                "dec_a1",
                "dec_b1",
                "inc_a2",
                "inc_b2",
                "chg_model",
                "dec_a3",
                "dec_b3",
                "inc_a4",
                "inc_b4",
                "dec_a4",
                "dec_b4",
                "inc_a3",
                "inc_b3",
                # "inc_c0",
                # "inc_c1",
                # "dec_c0",
                # "dec_c1"

            ])
