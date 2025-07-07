# XGBoost Qt C++ Wrapper

Этот проект предоставляет Qt-обёртку для XGBoost с двумя основными классами:
- `XGBRegressor` — для задач регрессии
- `XGBClassifier` — для задач классификации

## Зависимости
- Qt 5 (Core, Widgets)
- XGBoost C API (libxgboost)

## Классы и параметры

### 1. XGBRegressor

Класс для обучения и предсказания регрессионных моделей XGBoost.

**Конструктор:**
```cpp
XGBRegressor(const QMap<QString, QString>& params, QObject* parent = nullptr);
```
- `params` — карта параметров XGBoost (например, `n_iter`, `max_depth`, `eta`, `lambda`)
- `parent` — родительский QObject (обычно `this` для интеграции с Qt)

**Основные методы:**
- `void Fit(const QVector<QVector<double>>& X, const QVector<double>& y, float startProgressValue = 0.0f, float endProgressValue = 1.0f);`
  - `X` — матрица признаков (QVector строк, каждая строка — QVector<double>)
  - `y` — вектор целевых значений
  - `startProgressValue`, `endProgressValue` — значения для прогресс-бара (от 0.0 до 1.0)
- `QVector<double> Predict(const QVector<QVector<double>>& X);`
  - `X` — матрица признаков для предсказания
- `void SaveModel(const QString& filename);`
- `void LoadModel(const QString& filename);`
- `signals: void progress(float value);` — сигнал для отображения прогресса обучения

### 2. XGBClassifier

Класс для обучения и предсказания классификационных моделей XGBoost (мультикласс).

**Конструктор:**
```cpp
XGBClassifier(const QMap<QString, QString>& params, QObject* parent = nullptr);
```
- `params` — карта параметров XGBoost (например, `n_iter`, `max_depth`, `eta`, `lambda`)
- `parent` — родительский QObject

**Основные методы:**
- `void Fit(const QVector<QVector<double>>& X, const QVector<double>& y, float startProgressValue = 0.0f, float endProgressValue = 1.0f);`
  - `X` — матрица признаков
  - `y` — вектор меток классов (числовые значения)
- `void Fit(const QVector<QVector<double>>& X, const QVector<double>& y, const QVector<float>& stabilizer, float startProgressValue = 0.0f, float endProgressValue = 1.0f);`
  - `stabilizer` — дополнительный вектор весов для стабилизации (опционально)
- `QVector<double> Predict(const QVector<QVector<double>>& X);`
- `void SaveModel(const QString& filename);`
- `void LoadModel(const QString& filename);`
- `signals: void progress(float value);`

## Пример использования

```cpp
#include "xgbooster.hpp"

// --- Регрессия ---
QMap<QString, QString> params;
params["n_iter"] = "20";
params["max_depth"] = "4";
params["eta"] = "0.1";
params["lambda"] = "1";

QVector<QVector<double>> X = { {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0} };
QVector<double> y = {3.0, 5.0, 7.0};

XGBRegressor reg(params);
reg.Fit(X, y);
QVector<double> preds = reg.Predict(X);

// --- Классификация ---
QMap<QString, QString> cls_params;
cls_params["n_iter_"] = "10";
cls_params["max_depth"] = "3";
cls_params["eta"] = "0.2";
cls_params["lambda"] = "1";

QVector<QVector<double>> Xc = { {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0} };
QVector<double> yc = {0.0, 1.0, 2.0};

XGBClassifier clf(cls_params);
clf.Fit(Xc, yc);
QVector<double> preds_cls = clf.Predict(Xc);

// --- Сигнал прогресса (Qt) ---
connect(&reg, &XGBRegressor::progress, [](float value){
    qDebug() << "Regressor progress:" << value;
});
```

## Параметры XGBoost

- `n_iter` — количество boosting-итераций
- `max_depth` — максимальная глубина дерева
- `eta` — learning rate
- `lambda` — L2-регуляризация
- Для классификации автоматически выставляется `objective = multi:softmax`, для регрессии — `reg:squarederror`

## Сохранение и загрузка модели

```cpp
reg.SaveModel("reg.model");
reg.LoadModel("reg.model");
```

## Установка XGBoost
```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build && cd build
cmake ..
make
```
После сборки скомпилированная библиотека libxgboost.o будет лежать в xgboost/lib


## Программа ```xgbgui```
В корне репозитория лежит QMake проект ```xgbgui.pro```. Это QT GUI утилита, 
которая открывает csv-файл с данными и производит обучение на 66% данных, предказание 
на остальных. 
Для успешной сборки нужно положить в папку lib скомпированную библиотеку libxgboost.o,
а в include/xgboost - заголовочные файлы библиотеки XGBoost (лежат в xgboost/include/xgboost)
