from django.shortcuts import render

import math
import billiard as multiprocessing
import datetime
import requests
import operator
import csv
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from django.shortcuts import render
# import multiprocessing
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------ demand
list_print1 = ['Динамика уровня зарплат по годам: ', 'Динамика количества вакансий по годам: ',
               'Динамика уровня зарплат по годам для выбранной профессии: ',
               'Динамика количества вакансий по годам для выбранной профессии: ',
               'Уровень зарплат по городам (в порядке убывания): ',
               'Доля вакансий по городам (в порядке убывания): ']


class Solution:
    def __init__(self, path_to_file, name_vacancy):
        self.path_to_file = path_to_file
        self.name_vacancy = name_vacancy
        self.dynamics1 = {}
        self.dynamics2 = {}
        self.dynamics3 = {}
        self.dynamics4 = {}
        self.dynamics5 = {}
        self.dynamics6 = {}

    def split_by_year(self):
        data_of_file = pd.read_csv(self.path_to_file, low_memory=False)
        data_of_file["year"] = data_of_file["published_at"].apply(lambda x: x[:4])
        data_of_file = data_of_file.groupby("year")
        for year, data in data_of_file:
            data[["name", "salary_from", "salary_to", "salary_currency", "area_name", "published_at"]]. \
                to_csv(rf"templates/data/year_number_{year}.csv", index=False)

    def get_dynamics(self):
        self.get_dynamics_by_year_with_multiprocessing()
        self.get_dynamics_by_city()
        return self.dynamics1, self.dynamics2, self.dynamics3, self.dynamics4, self.dynamics5, self.dynamics6

    def get_statistic_by_year(self, file_csv):

        data_of_file = pd.read_csv(file_csv, low_memory=False)
        data_of_file["salary"] = data_of_file[["salary_from", "salary_to"]].mean(axis=1)
        data_of_file["salary"] = data_of_file["salary"].apply(lambda s: 0 if math.isnan(s) else int(s))
        data_of_file["published_at"] = data_of_file["published_at"].apply(lambda s: int(s[:4]))
        data_of_file_vacancy = data_of_file[data_of_file["name"].str.contains(self.name_vacancy, case=False)]

        return data_of_file["published_at"].values[0], [int(data_of_file["salary"].mean()), len(data_of_file),
                                                        int(data_of_file_vacancy["salary"].mean() if len(
                                                            data_of_file_vacancy) != 0 else 0),
                                                        len(data_of_file_vacancy)]

    def get_dynamics_by_year_with_multiprocessing(self):
        files = [rf"templates/data/{file_name}" for file_name in
                 os.listdir(rf"templates/data")]
        pool = multiprocessing.Pool(12)
        result = pool.starmap(self.get_statistic_by_year, [(file,) for file in files])
        pool.close()
        for year, data_dynamics in result:
            self.dynamics1[year] = data_dynamics[0]
            self.dynamics2[year] = data_dynamics[1]
            if data_dynamics[2] > data_dynamics[3]:
                self.dynamics3[year] = data_dynamics[2]
                self.dynamics4[year] = data_dynamics[3]
            elif data_dynamics[2] < data_dynamics[3]:
                self.dynamics3[year] = data_dynamics[3]
                self.dynamics4[year] = data_dynamics[2]



    def get_dynamics_by_city(self):
        data_of_file = pd.read_csv(self.path_to_file, low_memory=False)
        total = len(data_of_file)
        data_of_file["salary"] = data_of_file[["salary_from", "salary_to"]].mean(axis=1)
        data_of_file["count"] = data_of_file.groupby("area_name")["area_name"].transform("count")
        data_of_file = data_of_file[data_of_file["count"] > total * 0.01]
        data_of_file = data_of_file.groupby("area_name", as_index=False)
        data_of_file = data_of_file[["salary", "count"]].mean().sort_values("salary", ascending=False)
        data_of_file["salary"] = data_of_file["salary"].apply(lambda s: 0 if math.isnan(s) else int(s))

        self.dynamics5 = dict(zip(data_of_file.head(10)["area_name"], data_of_file.head(10)["salary"]))

        data_of_file = data_of_file.sort_values("count", ascending=False)
        data_of_file["count"] = round(data_of_file["count"] / total, 4)
        res = {}
        for k, v in self.dynamics5.items():
            if v > 150000:
                v = int(v * 0.14)
            res[k] = v
        self.dynamics5 = dict(sorted(res.items(), key=operator.itemgetter(1), reverse=True))
        self.dynamics6 = dict(zip(data_of_file.head(10)["area_name"], data_of_file.head(10)["count"]))

    def get_statistic(self):
        InputConnect(self.path_to_file, self.name_vacancy, self.dynamics1, self.dynamics2, self.dynamics3,
                     self.dynamics4, self.dynamics5, self.dynamics6)
        return self.dynamics1, self.dynamics2, self.dynamics3, self.dynamics4, self.dynamics5, self.dynamics6


class InputConnect:
    def __init__(self, path_to_file, name_vacancy, dynamics1, dynamics2, dynamics3, dynamics4, dynamics5,
                 dynamics6):
        self.path_to_file, self.name_vacancy = path_to_file, name_vacancy
        dynamics1, dynamics2, dynamics3, dynamics4, dynamics5, dynamics6 = dynamics1, dynamics2, dynamics3, dynamics4, dynamics5, dynamics6
        new_graphic = Report(self.name_vacancy, dynamics1, dynamics2, dynamics3, dynamics4, dynamics5, dynamics6)
        new_graphic.generate_image_demand()
        new_graphic.generate_image_geo()


def dict_sort(dict):
    arr = sorted(dict.items())
    dict2 = {}
    for cort in arr:
        dict2[cort[0]] = cort[1]
    return dict2


class Report:
    def __init__(self, name_vacancy, dynamics1, dynamics2, dynamics3, dynamics4, dynamics5, dynamics6):
        self.name_vacancy = name_vacancy
        self.dynamics1 = dict_sort(dynamics1)
        self.dynamics2 = dict_sort(dynamics2)
        self.dynamics3 = dict_sort(dynamics3)
        self.dynamics4 = dict_sort(dynamics4)
        self.dynamics6 = dynamics6
        self.dynamics5 = dynamics5


    def generate_image_demand(self):
        x = np.arange(len(self.dynamics1.keys()))
        width = 0.35

        fig, axs = plt.subplots(1, 2)
        axs[0].bar(x - width / 2, self.dynamics1.values(), width, label='средняя з/п')
        axs[0].bar(x + width / 2, self.dynamics3.values(), width, label='з/п C++ программист')
        plt.rcParams['font.size'] = '8'
        for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels()):
            label.set_fontsize(7)
        axs[0].set_title('Уровень зарплат по годам')
        axs[0].set_xticks(x, self.dynamics1.keys(), rotation=90)
        axs[0].grid(axis='y')
        axs[0].legend(fontsize=7)

        axs[1].bar(x - width / 2, self.dynamics2.values(), width, label='количество вакансий')
        axs[1].bar(x + width / 2, self.dynamics4.values(), width,
                   label='количество вакансий C++ программист')
        for label in (axs[1].get_xticklabels() + axs[1].get_yticklabels()):
            label.set_fontsize(7)
        axs[1].set_title('Количество вакансий по годам')
        axs[1].set_xticks(x, self.dynamics2.keys(), rotation=90)
        axs[1].grid(axis='y')
        axs[1].legend(fontsize=7)
        fig.tight_layout()

        plt.tight_layout()
        plt.savefig(r'C:\Users\artyo\sites\ProjectPython\static\graph1.png', dpi=300)

    def generate_image_geo(self):
        fig, axs = plt.subplots(ncols=2, nrows=1)
        fig.tight_layout()
        areas = []
        for area in self.dynamics5.keys():
            areas.append(str(area).replace(' ', '\n').replace('-', '-\n'))
        y_pos = np.arange(len(areas))
        performance = self.dynamics5.values()
        error = np.random.rand(len(areas))
        axs[0].barh(y_pos, performance, xerr=error, align='center')
        for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels()):
            label.set_fontsize(7)
        axs[0].set_yticks(y_pos, labels=areas, size=7)
        axs[0].invert_yaxis()
        axs[0].grid(axis='x')
        axs[0].set_title('Уровень зарплат по городам')

        val = list(self.dynamics6.values()) + [1 - sum(list(self.dynamics6.values()))]
        k = list(self.dynamics6.keys()) + ['Другие']
        axs[1].pie(val, labels=k, startangle=150)
        axs[1].set_title('Доля вакансий по городам')

        plt.tight_layout()
        plt.savefig(r'C:\Users\artyo\sites\ProjectPython\static\graph2.png', dpi=300)


def demand(request):
    filename = r'C:\Users\artyo\sites\ProjectPython\templates\data\vacancies_with_skills.csv'
    name_vacancy = "C+"
    solve = Solution(filename, name_vacancy)
    solve.split_by_year()
    solve.get_dynamics()
    dynamics1, dynamics2, dynamics3, dynamics4, dynamics5, dynamics6 = solve.get_statistic()
    dynamics = []

    for year in dynamics2.keys():
        dynamics.append([year, dynamics1[year], dynamics2[year], dynamics3[year], dynamics4[year]])

    dynamics.sort()

    data = {'name': name_vacancy,
            'path': r'C:\Users\artyo\sites\ProjectPython\static\graph1.png',
            'val0': dynamics[0],
            'val1': dynamics[1],
            'val2': dynamics[2],
            'val3': dynamics[3],
            'val4': dynamics[4],
            'val5': dynamics[5],
            'val6': dynamics[6],
            'val7': dynamics[7],
            'val8': dynamics[8],
            'val9': dynamics[9],
            'val10': dynamics[10],
            'val11': dynamics[11],
            'val12': dynamics[12],
            'val13': dynamics[13],
            'val14': dynamics[14],
            'val15': dynamics[15],
            'val16': dynamics[16],
            'val17': dynamics[17],
            'val18': dynamics[18],
            'val19': dynamics[19],
            "dinamics": dynamics,
            }

    return render(request, "demand.html", context=data)


def main(request):
    return render(request, "main.html")


def geography(request):
    filename = r'C:\Users\artyo\sites\ProjectPython\templates\data\vacancies_with_skills.csv'
    name_vacancy = "C+"
    solve = Solution(filename, name_vacancy)
    solve.get_dynamics()
    dynamics1, dynamics2, dynamics3, dynamics4, dynamics5, dynamics6 = solve.get_statistic()
    for key in dynamics6:
        dynamics6[key] = round(dynamics6[key] * 100, 2)
    res1 = []
    res2 = []
    res3 = []
    for k, v in dynamics5.items():
        res1.append(str(k) + ":" + str(v) + ":")

    for k2, v2 in dynamics6.items():
        res2.append(str(k2) + ":" + str(v2) + ":")

    for i in range(len(res1)):
        res3.append(res1[i] + res2[i])

    def get_vals(res_arr):
        val_arr = res_arr.split(":")
        res = []
        for i, el in enumerate(val_arr):
            if i == 4:
                break
            else:
                res.append(el)
        del val_arr[0:4]
        return res

    val1 = get_vals(res3[0])
    val2 = get_vals(res3[1])
    val3 = get_vals(res3[2])
    val4 = get_vals(res3[3])
    val5 = get_vals(res3[4])
    val6 = get_vals(res3[5])
    val7 = get_vals(res3[6])
    val8 = get_vals(res3[7])
    val9 = get_vals(res3[8])
    val10 = get_vals(res3[9])
    data = {
        'dynamics5': dynamics5.items(),
        'dynamics6': dynamics6.items(),
        'val1' : val1,
        'val2': val2,
        'val3': val3,
        'val4': val4,
        'val5': val5,
        'val6': val6,
        'val7': val7,
        'val8': val8,
        'val9': val9,
        'val10': val10

    }
    return render(request, "geography.html", context=data)


# ----------------------------------------


vacancy_choices = '|'.join(['C+'])
data = pd.read_csv(r'C:\Users\artyo\sites\ProjectPython\templates\data\vacancies_with_skills.csv',
                   usecols=['name', 'key_skills', 'published_at'])

data = data[data['name'].str.contains(vacancy_choices, case=False)]
data = data[data['key_skills'].notnull()]
data['year'] = data['published_at'].apply(lambda x: x[:4])
groups = data.groupby(['year'])


def write_group_to_csv(group):
    year, data = group
    data.to_csv(f'skills_by_city/key_skills_by_{year}.csv', index=False)


thread_pool = []

for group in groups:
    thread = threading.Thread(target=write_group_to_csv, args=(group,))
    thread.start()
    thread_pool.append(thread)

for thread in thread_pool:
    thread.join()


class Skill:
    def __init__(self, name, amount):
        self.name = name
        self.amount = amount

    def __repr__(self):
        return f'{self.name}: {self.amount}'


def analyze_year(file_name):
    all_skills = []
    with open(file_name, 'r', encoding='utf-8-sig') as skills:
        rows = csv.reader(skills)
        next(rows)
        for row in rows:
            s = row[1]
            for i in s.split('\n'):
                all_skills.append(i)
    DATA = pd.DataFrame({'skills': all_skills}).value_counts().head(10).to_dict()
    date = file_name[-8:-4]
    return date, [(skill[0], amount) for skill, amount in DATA.items()]


def skills(request):
    file_names = [f'skills_by_city/{file}' for file in os.listdir('skills_by_city')]
    skills_arr = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as ex:
        res = ex.map(analyze_year, file_names)
        data = {"res": sorted(res)}
    return render(request, "skills.html", context=data)


# ------------------------


def clean_vacancy(vacancy):
    # vacancy['area'] = vacancy['area']['name'] if vacancy['area'].__contains__('name') else 'Нет данных'
    if vacancy['salary']['from'] != None and vacancy['salary']['to'] != None and vacancy['salary']['from'] != \
            vacancy['salary']['to']:
        vacancy[
            'salary'] = f"от {'{0:,}'.format(vacancy['salary']['from']).replace(',', ' ')} до {'{0:,}'.format(vacancy['salary']['to']).replace(',', ' ')} {vacancy['salary']['currency']}"
    elif vacancy['salary']['from'] != None:
        vacancy[
            'salary'] = f"{'{0:,}'.format(vacancy['salary']['from']).replace(',', ' ')} {vacancy['salary']['currency']}"
    elif vacancy['salary']['to'] != None:
        vacancy[
            'salary'] = f"{'{0:,}'.format(vacancy['salary']['to']).replace(',', ' ')} {vacancy['salary']['currency']}"
    else:
        vacancy['salary'] = 'Нет данных'
    vacancy['key_skills'] = ', '.join(map(lambda x: x['name'], vacancy['key_skills']))
    return vacancy


def get_vacancies():
    try:
        data = []
        info = requests.get(r'https://api.hh.ru/vacancies?text=%22C++%22&specialization=1&per_page=100',
                            params={"date_from": "2022-12-23T12:00:00+0300", "date_to": "2022-12-23T23:59:00+0300"}).json()
        for row in info['items']:
            if row['name'].lower().__contains__('c+') and not row['salary'] is None:
                data.append({'id': row['id'], 'published_at': row['published_at']})
        data = sorted(data, key=lambda x: x['published_at'])
        vacancies = []
        for vacancy in data[len(data) - 10:]:
            vacancies.append(clean_vacancy(requests.get(f'https://api.hh.ru/vacancies/{vacancy["id"]}').json()))
        return vacancies
    except Exception as e:
        print(e)
        print(datetime.datetime.now())
        return []


def last_vacansies(request):
    return render(request, "last_vacansies.html", context={"vacansies": get_vacancies(), })
