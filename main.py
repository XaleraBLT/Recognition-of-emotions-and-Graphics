import logging
import webbrowser
import os

# logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
# logging.basicConfig(level=logging.INFO)
import datetime
import json
import os
import time

from fer import FER
import cv2
import threading
import matplotlib.pyplot as plt
from fer.utils import draw_annotations

cap = cv2.VideoCapture(0)
test_images = []
emo_detector = FER(mtcnn=True)


def emotions_get(img):
    captured_emotions = emo_detector.detect_emotions(img)

    return captured_emotions

def get_image():
    start = time.time()
    global run
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('saves/video.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS) / 3), (int(cap.get(3)), int(cap.get(4))))
    result = [0, 0, 0, 0, 0, 0, 0]
    result_prom = result.copy()
    try:
        emotions = open("saves/emotions.blt", "r+", encoding="utf-8")
    except:
        emotions = {}
    stress = 0
    stress_streak = 1

    while run:
        ret, frame = cap.read()
        if not ret:
            print("Не возможно инициализировать камеру!\nОстановлено")
            run = False
            out.release()
            break
        frameOpencvDnn = frame.copy()
        try:
            res = emotions_get(frameOpencvDnn)
            for i in range(len(result)):
                result[i] += list(res[0]['emotions'].values())[i]
                result_prom[i] += list(res[0]['emotions'].values())[i]

            current_time = datetime.datetime.now()
            emotions[f"{current_time.hour}:{current_time.minute}:{current_time.second}"] = {
                "злость": sum([res[i]['emotions']['angry'] for i in range(len(res))]),
                "отвращение": sum([res[i]['emotions']['disgust'] for i in range(len(res))]),
                "тревожность": sum([res[i]['emotions']['fear'] for i in range(len(res))]),
                "радость": sum([res[i]['emotions']['happy'] for i in range(len(res))]),
                "грусть": sum([res[i]['emotions']['sad'] for i in range(len(res))]),
                "удивление": sum([res[i]['emotions']['surprise'] for i in range(len(res))]),
                "нейтральность": sum([res[i]['emotions']['neutral'] for i in range(len(res))])
            }

            image = draw_annotations(frameOpencvDnn, res)
            image = cv2.resize(image.copy(), (0, 0), fx=1.5, fy=1.5)
            cv2.imshow('emotions', image)


            work_time = datetime.timedelta(seconds=round(time.time() - start))
            with open("saves/analiz.blt", "w+", encoding="utf-8") as file:
                json.dump({"result": result, "time_work": str(work_time)}, file, ensure_ascii=False, indent=4)

            with open('saves/emotions.blt', 'w+', encoding="utf-8") as emo:
                json.dump(emotions, emo, indent=4, ensure_ascii=False)

            cv2.imshow('emotions', image)
            out.write(image)
            cv2.waitKey(1)
        except IndexError:

            cv2.waitKey(1)


    out.release()
    return result, time.time()

def start_capture(time_start):
    global run
    run = True
    result, time_work = get_image()
    work_time = datetime.timedelta(seconds=round(time_work - time_start))
    with open("saves/analiz.blt", "w+", encoding="utf-8") as file:
        json.dump({"result": result, "time_work": str(work_time)}, file, ensure_ascii=False, indent=4)

def main():
    global run
    while True:
        command = input("Введите команду: ")
        if command == "/stop":
            run = False
            print("Остановлено")
        elif command == "/start":

            if not run:
                try:
                    os.remove('saves/video.mp4')
                except:
                    pass
                try:
                    os.remove('saves/emotions.blt')
                except:
                    pass
                try:
                    os.remove('saves/analiz.blt')
                except:
                    pass

                threading.Thread(target=start_capture, args=(time.time(),)).start()
            else:
                print("Уже запущено")
        elif command == "/diogram-pie":
            data_names = {'злость': 'red', 'отвращение': 'green', 'тревожность': 'purple', 'радость': 'yellow',
                          'грусть': 'blue', 'удивление': 'orange', 'нейтральность': 'grey'}

            if run:
                print("Остановите анализ!")
            else:
                with open("saves/analiz.blt", "r+", encoding="utf-8") as file:
                    data = json.loads(file.read())

                    try:
                        explode = (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15)
                        fig, ax = plt.subplots()
                        ax.pie(data["result"], labels=list(data_names.keys()), autopct='%1.1f%%', shadow=True, explode=explode,
                               wedgeprops={'lw': 1, 'ls': '--', 'edgecolor': "k"}, rotatelabels=True, colors=list(data_names.values()))
                        ax.axis("equal")

                        plt.show()
                    except:
                        print("Запустите анализ!")
        elif command == '/diogram-full':
            if run:
                print("Остановите анализ!")
            else:
                try:
                    emote = json.loads(open('saves/emotions.blt', 'r+', encoding="utf-8").read())
                    data_names = {'злость': 'red', 'отвращение': 'green', 'страх': 'black', 'радость': 'yellow',
                                  'грусть': 'blue', 'удивление': 'orange', 'нейтральность': 'grey'}

                    dates = emote.keys()
                    for name in data_names.keys():
                        values = []
                        for date in dates:
                            values.append(float(emote[date][name]) * 100)
                        plt.plot(list(dates), values, color=data_names.get(name), label=str(name), marker='o')

                    plt.title('Анализ настроения')
                    plt.ylabel('Уровень настроения, %')
                    plt.xlabel('Время, Ч:м:c')

                    plt.show()
                except Exception as e:
                    print(e)
                    print("Запустите анализ!")
        else:
            print("Не распознана команда.",
                  "Список текущих команд:",
                  " - /start - запустить",
                  " - /stop - остановить",
                  " - /diogram-pie - сводка ввиде круговой диаграммы",
                  " - /diogram-full - сводка ввиде классической диаграммы", sep="\n")
if __name__ == "__main__":
    run = False

    main()
    cap.release()
    cv2.destroyAllWindows()




