Для запуска обученния - прописать в терминале:
    python ex4_train.py \
        --n_way 220 \
        --n_support 10 \
        --n_query 10 \
        --max_epoch 5 \
        --epoch_size 2000 \
        --lr 0.001 
    

Для запуска тестирования - прописать в терминале:
    python ex4_test.py --checkpoint trained_model_v1.pt


Epoch 1 -- Loss: 3.0948 Acc: 0.3506
Epoch 2 -- Loss: 2.4375 Acc: 0.4597
Epoch 3 -- Loss: 2.2549 Acc: 0.4920
Epoch 4 -- Loss: 2.1726 Acc: 0.5069
Epoch 5 -- Loss: 2.1350 Acc: 0.5137



Описание состава команды

Андреев Эдуард Дмитриевич
- Тимлид
- Выполнил ex3 - реализовал Prototypical Networks - дописал set_forward_loss.
- Выполнил ex4 - добавил логирование neptune в процесс обучения.
- Выполнил ex6_1 -  обучил модель методом SimCLR, а затем воспользовался обученным энкодером для решения задачи классификации.

Васильев Никита Васильевич
- Выполнил ex5 - добавил функцию для визуализации предсказания с тестирования.
- Выполнил ex6_2 - сохранил графики loss и accuracy. Добавил логирование в процесс обучения в ex6.
- Обучил модель и разделил код на .py файлы.

Шарапов Ильдар Маратович
- Выполнил ex1 - реализовал функцию чтения данных.
- Выполнил ex2 - реализовал энкодер на основе сверточной нейроннй сети.
