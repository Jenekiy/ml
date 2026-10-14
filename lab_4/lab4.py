import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=69, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

C_arr = np.logspace(-3, 3, 7)
log_err, svm_err = [], []

for C in C_arr:
    # Логистическая регрессия (с L2 регуляризацией)
    log = LogisticRegression(C=C, max_iter=10000, random_state=69)
    log.fit(X_train, y_train)
    log_err.append(accuracy_score(y_test, log.predict(X_test_scaled)))

    # Linear SVM (с L2 регуляризацией)
    svm = LinearSVC(C=C, max_iter=10000, dual=True, random_state=69)
    svm.fit(X_train, y_train)
    svm_err.append(accuracy_score(y_test, svm.predict(X_test_scaled)))

true_C_log = C_arr[np.argmax(log_err)]
true_C_svm = C_arr[np.argmax(svm_err)]

print(f"LogisticRegression: L2 = {max(log_err):.4f}, C = {true_C_log}")
print(f"SVC:          L2 = {max(svm_err):.4f}, C = {true_C_svm}")

plt.figure(figsize=(15, 4))

# График: Зависимость качества от коэффициента C
plt.subplot(1, 4, 1)
plt.semilogx(C_arr, log_err, label="Лог. Регрессия", marker="o")
plt.semilogx(C_arr, svm_err, label="SVM", marker="s")
plt.xlabel("Параметр регуляризации C")
plt.ylabel("L2 на тесте")
plt.title("Зависимость качества от C")
plt.legend()
plt.grid(True)

# График: Поведение весов признаков при разной силе регуляризации(LOG)
plt.subplot(1, 4, 2)
C_sub = [0.001, 0.1, 100.0]  # Слабая, оптимальная и избыточная регуляризация
for C in C_sub:
    model = LogisticRegression(C=C, max_iter=10000, random_state=69).fit(X_train, y_train)
    plt.plot(model.coef_[0], label=f"log (C={C})", marker="o", linestyle="--")
plt.xlabel("Индекс признака")
plt.ylabel("Вес (Коэффициент)")
plt.title("Веса признаков при разных C")
plt.legend()
plt.grid(True)

# График: Поведение весов признаков при разной силе регуляризации(SVM)
plt.subplot(1, 4, 3)
C_sub = [0.001, 0.1, 100.0]  # Слабая, оптимальная и избыточная регуляризация
for C in C_sub:
    model = LinearSVC(C=C, max_iter=10000, dual=True, random_state=69).fit(X_train, y_train)
    plt.plot(model.coef_[0], label=f"SVM (C={C})", marker="o", linestyle="--")
plt.xlabel("Индекс признака")
plt.ylabel("Вес (Коэффициент)")
plt.title("Веса признаков при разных C")
plt.legend()
plt.grid(True)


# График: Разделяющие прямые по двум главным признакам
final_model = LogisticRegression(C=true_C_log, max_iter=10000, random_state=69).fit(X_train, y_train)
top_two_features = np.argsort(np.abs(final_model.coef_[0]))[-2:]

X_train_2d = X_train[:, top_two_features]
log_2d = LogisticRegression(C=true_C_log, random_state=69).fit(X_train_2d, y_train)
svm_2d = LinearSVC(C=true_C_svm, dual=True, random_state=69).fit(X_train_2d, y_train)

plt.subplot(1, 4, 4)
ax = plt.gca()
x_lim, y_lim = (-3, 3), (-3, 3)
x_point, y_point = np.meshgrid(np.linspace(*x_lim, 100), np.linspace(*y_lim, 100))

Z_log = log_2d.decision_function(np.c_[x_point.ravel(), y_point.ravel()]).reshape(x_point.shape)
line_log = ax.contour(x_point, y_point, Z_log, levels=[0], colors="blue", linestyles="-")

Z_svm = svm_2d.decision_function(np.c_[x_point.ravel(), y_point.ravel()]).reshape(x_point.shape)
line_svm = ax.contour(x_point, y_point, Z_svm, levels=[0], colors="orange", linestyles="--")

ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", alpha=0.5)
ax.set_xlabel(f"{data.feature_names[top_two_features[0]]} (scaled)")
ax.set_ylabel(f"{data.feature_names[top_two_features[1]]} (scaled)")
ax.set_title("Разделяющая линия")

lines = [line_log.legend_elements()[0][0], line_svm.legend_elements()[0][0]]
labels = ["Лог. Регрессия", "SVM"]
plt.legend(lines, labels)

plt.tight_layout()
plt.show()