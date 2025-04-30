class Maze:
    def __init__(self, ncol = 9, nrow = 6, model = "basic"):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标
        self.time_counter = 0
        self.counter_epi = 0
        self.model = model
        if model == "basic":
            self.barrier = [[2, 2], [2, 3], [2, 4], [5, 1], [7, 3], [7, 4], [7, 5]]
            self.start = [0, 3]
            self.end = [8, 5]
        if model == "blocking":
            self.barrier = [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2]]
            self.start = [3, 0]
            self.end = [8, 5]
        if model == "cut":
            self.barrier = [[8, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2]]
            self.start = [3, 0]
            self.end = [8, 5]

    def step(self, action):  # 外部调用这个函数来改变当前位置
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        newx = min(self.ncol - 1, max(0, self.x + change[action][0]))
        newy = min(self.nrow - 1, max(0, self.y + change[action][1]))
        if [newx, newy] not in self.barrier:
            self.x = newx
            self.y = newy
        next_state = self.y * self.ncol + self.x
        reward = 0
        done = False
        self.time_counter += 1
        self.counter_epi += 1
        if self.x==self.end[0] and self.y==self.end[1]:  # 下一个位置在终点
            done = True
            reward = 1
            self.counter_epi = 0
        if self.counter_epi == 10000:
            print("don't reach the end in 10000 steps")
        if self.time_counter == 5000 and self.model == "blocking":
            self.barrier = [[8, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2]]
        if self.time_counter == 3000 and self.model == "cut":
            self.barrier = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2]]
        return next_state, reward, done

    def reset(self):  # 回归初始状态,起点在左上角
        self.counter_epi = 0
        self.x = self.start[0]
        self.y = self.start[1] 
        return self.y * self.ncol + self.x