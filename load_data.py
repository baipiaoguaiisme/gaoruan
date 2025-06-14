import torch
import torch.utils.data as data
import numpy as np


class getReader():
    def __init__(self, path):
        self.path = path

    def readData(self):

        problem_list = []
        ans_list = []
        split_char = ','

        read = open(self.path, 'r')
        for index, line in enumerate(read):
            if index % 3 == 0:
                pass

            elif index % 3 == 1:
                problems = line.strip().split(split_char)
                # 由于列表problems每个元素都是char 需要变为int
                problems = list(map(int, problems))
                problem_list.append(problems)

            elif index % 3 == 2:
                ans = line.strip().split(split_char)
                # 由于列表ans每个元素都是char 需要变为int
                ans = list(map(float, ans))
                ans = [int(x) for x in ans]
                ans_list.append(ans)

        read.close()
        return problem_list, ans_list


class KT_Dataset(data.Dataset):
    def __init__(self, problem_max, problem_list, skill_list, ans_list, min_problem_num, max_problem_num):
        self.problem_max = problem_max
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num
        self.problem_list, self.ans_list = [], []
        self.skill_list = []
        # 个人定义，少于 min_problem_num 丢弃
        # 根据论文 多于 max_problem_num  的分成多个 max_problem_num 
        for (problem, ans, skill) in zip(problem_list, ans_list, skill_list):
            num = len(problem)
            if num < min_problem_num:
                continue
            elif num > max_problem_num:
                segment = num // max_problem_num
                now_problem = problem[num - segment * max_problem_num:]
                now_ans = ans[num - segment * max_problem_num:]
                now_skill = skill[num - segment * max_problem_num:]

                if num > segment * max_problem_num:
                    self.problem_list.append(problem[:num - segment * max_problem_num])
                    self.ans_list.append(ans[:num - segment * max_problem_num])
                    self.skill_list.append(skill[:num - segment * max_problem_num])

                for i in range(segment):
                    item_problem = now_problem[i * max_problem_num:(i + 1) * max_problem_num]
                    item_ans = now_ans[i * max_problem_num:(i + 1) * max_problem_num]
                    self.problem_list.append(item_problem)
                    self.ans_list.append(item_ans)
                    self.skill_list.append(now_skill[i * max_problem_num:(i + 1) * max_problem_num])

            else:
                item_problem = problem
                item_ans = ans
                self.problem_list.append(item_problem)
                self.ans_list.append(item_ans)
                self.skill_list.append(skill)


    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        now_problem = self.problem_list[index]
        now_problem = np.array(now_problem)
        now_skill = self.skill_list[index]
        now_ans = self.ans_list[index]
        # 由于需要统一格式
        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=int)

        use_skill = np.zeros(self.max_problem_num, dtype=int)

        use_mask = np.zeros(self.max_problem_num, dtype=int)

        num = len(now_problem)
        use_problem[-num:] = now_problem  # 将problem插入到列表尾部
        use_ans[-num:] = now_ans
        use_skill[-num:] = now_skill

        next_ans = use_ans[1:]
        next_problem = use_problem[1:]
        next_skill = use_skill[1:]

        last_ans = use_ans[:-1]
        last_problem = use_problem[:-1]
        last_skill = use_skill[:-1]

        mask = np.zeros(self.max_problem_num - 1, dtype=int)
        mask[-num + 1:] = 1

        use_mask[-num:] = 1
        last_mask = use_mask[:-1]
        next_mask = use_mask[1:]

        last_problem = torch.from_numpy(last_problem).to(device).long()
        next_problem = torch.from_numpy(next_problem).to(device).long()
        last_ans = torch.from_numpy(last_ans).to(device).long()
        next_ans = torch.from_numpy(next_ans).to(device).float()

        last_skill = torch.from_numpy(last_skill).to(device).long()
        next_skill = torch.from_numpy(next_skill).to(device).long()

        return last_problem, last_skill, last_ans, next_problem, next_skill, next_ans, torch.tensor(mask == 1).to(
            device)


def getLoader(problem_max, pro_path, skill_path, batch_size, is_train, min_problem_num, max_problem_num):
    problem_list, ans_list = getReader(pro_path).readData()  # problem_list shape [batch,problem size]
    skill_list, ans_list = getReader(skill_path).readData()  # skill_list shape [batch,problem size]
    dataset = KT_Dataset(problem_max, problem_list, skill_list, ans_list, min_problem_num, max_problem_num)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return loader


if __name__ == '__main__':
    problem_list, skill_list, ans_list = getLoader(168, './data/ASSIST09/train_question.txt',
                                                   './data/ASSIST09/train_skill.txt',
                                                   80, True, 3, 200)
    dataset = KT_Dataset(168, problem_list, skill_list, ans_list, 3, 200)
    data = dataset.__getitem__(0)
    print(data)
    for i in data:
        print(i.shape)
