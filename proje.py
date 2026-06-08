
import json
import os
import sys
import time

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "DisabledHAR_dataset_v1.csv")
OUT = os.path.join(BASE, "outputs")
FIG = os.path.join(OUT, "figures")
MET = os.path.join(OUT, "metrics")
for _d in (OUT, FIG, MET):
    os.makedirs(_d, exist_ok=True)

TARGET, GROUP, DISABLED = "label", "UserID", "disabled"
ACCEL = ["accelerometerAccelerationX", "accelerometerAccelerationY", "accelerometerAccelerationZ"]
GYRO = ["gyroRotationX", "gyroRotationY", "gyroRotationZ"]
LOC = [
    "locationLatitude",
    "locationLongitude",
    "locationAltitude",
    "locationSpeed",
    "locationCourse",
    "locationVerticalAccuracy",
]
IMU = [
    "accelerometerAccelerationX",
    "accelerometerAccelerationY",
    "accelerometerAccelerationZ",
    "gyroRotationX",
    "gyroRotationY",
    "gyroRotationZ",
    "magnetometerX",
    "magnetometerY",
    "magnetometerZ",
    "motionYaw",
    "motionRoll",
    "motionPitch",
    "motionRotationRateX",
    "motionRotationRateY",
    "motionRotationRateZ",
    "motionUserAccelerationX",
    "motionUserAccelerationY",
    "motionUserAccelerationZ",
    "motionGravityX",
    "motionGravityY",
    "motionGravityZ",
    "motionMagneticFieldX",
    "motionMagneticFieldY",
    "motionMagneticFieldZ",
]
LABELS = ["Walking", "jogging", "Upstairs", "Downstairs", "Standing", "Sitting"]
LABEL_TR = {
    "Walking": "Yürüme",
    "jogging": "Koşu",
    "Upstairs": "Rampa çıkış",
    "Downstairs": "Rampa iniş",
    "Standing": "Ayakta",
    "Sitting": "Oturma",
}
RS = 42
N_TREES = 120

if __name__ == "__main__":
    _cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if _cmd not in ("explore", "eda", "train", "all"):
        print(__doc__)
        print("Geçersiz argüman. Kullanım: python proje.py  [explore|eda|train|all]")
        sys.exit(1)

    _do_ex = _cmd in ("explore", "all")
    _do_eda = _cmd in ("eda", "all")
    _do_tr = _cmd in ("train", "all")

    if _do_ex:
        if not os.path.isfile(CSV_PATH):
            sys.exit("CSV bulunamadı: " + CSV_PATH + "\nMendeley DOI: 10.17632/r78dn3f6nc.4")
        print("Yükleniyor:", CSV_PATH)
        _df = pd.read_csv(CSV_PATH)
        print(" ", len(_df), "satır ×", _df.shape[1], "sütun")
        print(_df.dtypes)
        print("\nlabel:\n", _df[TARGET].value_counts())
        print("\ndisabled:\n", _df[DISABLED].value_counts())
        print("\nUserID sayısı:", _df[GROUP].nunique())
        print("Eksik:", int(_df.isna().sum().sum()))

    if _do_eda:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})
        if not os.path.isfile(CSV_PATH):
            sys.exit("CSV bulunamadı: " + CSV_PATH)
        _df = pd.read_csv(CSV_PATH)
        _order = LABELS
        _xt = [LABEL_TR[x] for x in _order]
        _bw = 0.35

        _c = _df[TARGET].value_counts().reindex(_order)
        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.bar(_xt, _c.values, color="#2c7fb8")
        _ax.set_title("Aktivite dağılımı")
        _ax.set_ylabel("Örnek sayısı")
        plt.setp(_ax.get_xticklabels(), rotation=20, ha="right")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "eda_01_aktivite_dagilimi.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: eda_01_aktivite_dagilimi.png")

        _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
        _dvc = _df[DISABLED].value_counts().sort_index()
        _axes[0].pie(_dvc.values, labels=["Engelsiz", "Engelli"], autopct="%1.1f%%", startangle=90)
        _axes[0].set_title("Engelli / engelsiz")
        _ct = pd.crosstab(_df[TARGET], _df[DISABLED]).reindex(_order)
        _x = np.arange(len(_order))
        _axes[1].bar(_x - _bw / 2, _ct[0], _bw, label="Engelsiz")
        _axes[1].bar(_x + _bw / 2, _ct[1], _bw, label="Engelli")
        _axes[1].set_xticks(_x)
        _axes[1].set_xticklabels(_xt, rotation=20, ha="right")
        _axes[1].legend()
        _axes[1].set_title("Aktivite × engelli kırılımı")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "eda_02_engelli_dagilimi.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: eda_02_engelli_dagilimi.png")

        _pu = _df.groupby(GROUP).agg(n=(TARGET, "size"), dis=(DISABLED, "first"))
        _fig, _ax = plt.subplots(figsize=(9, 4))
        _col = ["#fc8d59" if v == 1 else "#99d8c9" for v in _pu["dis"]]
        _ax.bar(_pu.index.astype(str), _pu["n"], color=_col)
        _ax.set_title("Kişi başına örnek (turuncu=engelli)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "eda_03_kisi_basina_ornek.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: eda_03_kisi_basina_ornek.png")

        _corr = _df[IMU].corr()
        _fig, _ax = plt.subplots(figsize=(10, 8.5))
        _im = _ax.imshow(_corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        _ax.set_xticks(range(len(IMU)))
        _ax.set_yticks(range(len(IMU)))
        _ax.set_xticklabels(IMU, rotation=90, fontsize=6)
        _ax.set_yticklabels(IMU, fontsize=6)
        _ax.set_title("IMU Pearson korelasyonu")
        _fig.colorbar(_im, ax=_ax, fraction=0.035)
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "eda_04_korelasyon_imu.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: eda_04_korelasyon_imu.png")

        _df["_acc_mag"] = np.sqrt(_df[ACCEL[0]] ** 2 + _df[ACCEL[1]] ** 2 + _df[ACCEL[2]] ** 2)
        _df["_gyro_mag"] = np.sqrt(_df[GYRO[0]] ** 2 + _df[GYRO[1]] ** 2 + _df[GYRO[2]] ** 2)
        _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
        for _ax, _col, _tit in zip(
            _axes,
            ["_acc_mag", "_gyro_mag"],
            ["İvmeölçer |a| (g)", "Jiroskop |ω| (rad/s)"],
        ):
            _data = [_df.loc[_df[TARGET] == _lab, _col].values for _lab in _order]
            _bp = _ax.boxplot(_data, tick_labels=_xt, showfliers=False, patch_artist=True)
            for _patch in _bp["boxes"]:
                _patch.set_facecolor("#a6bddb")
            _ax.set_title(_tit)
            plt.setp(_ax.get_xticklabels(), rotation=22, ha="right", fontsize=9)
        _fig.suptitle("Aktivite imzaları: türetilmiş büyüklükler", fontsize=12)
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "eda_05_sensor_buyukluk_box.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: eda_05_sensor_buyukluk_box.png")

        _samp = _df.sample(min(15000, len(_df)), random_state=RS)
        _fig, _ax = plt.subplots(figsize=(6, 5))
        _sc = _ax.scatter(_samp["locationLongitude"], _samp["locationLatitude"], c=_samp[GROUP], s=3, alpha=0.5)
        _ax.set_title("GPS örnekleri (konum ≠ aktivite; sızıntı riski)")
        _ax.set_xlabel("Boylam")
        _ax.set_ylabel("Enlem")
        _fig.colorbar(_sc, ax=_ax, label="UserID")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "eda_06_konum_sizintisi.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: eda_06_konum_sizintisi.png")

        _fig, _ax = plt.subplots(figsize=(8, 4))
        _vc = _df[TARGET].value_counts()
        _ax.pie(_vc.values, labels=[LABEL_TR.get(i, i) for i in _vc.index], autopct="%1.1f%%", startangle=140)
        _ax.set_title("Aktivite oranları (pasta)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "eda_07_aktivite_pasta.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: eda_07_aktivite_pasta.png")

        _txt = "Örnek: {:,}\nIMU sütun: {}\nKişi: {}\nEksik: {}\n".format(
            len(_df), len(IMU), _df[GROUP].nunique(), int(_df[IMU + LOC].isna().sum().sum())
        )
        with open(os.path.join(MET, "00_veri_ozeti.txt"), "w", encoding="utf-8") as _f:
            _f.write(_txt)
        print(_txt)
        print("EDA bitti.")

    if _do_tr:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_recall_fscore_support,
            roc_curve,
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.preprocessing import label_binarize
        from sklearn.tree import plot_tree

        plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 10})
        _bw = 0.36
        _t0 = time.time()
        if not os.path.isfile(CSV_PATH):
            sys.exit("CSV bulunamadı: " + CSV_PATH)
        print("Yükleniyor:", CSV_PATH)
        _df = pd.read_csv(CSV_PATH)
        print(" ", len(_df), "satır — Rastgele Orman, ağaç sayısı:", N_TREES)
        _X = _df[IMU].values
        _y = _df[TARGET].values
        _g = _df[GROUP].values
        _dis = _df[DISABLED].values

        _rf = RandomForestClassifier(
            n_estimators=N_TREES,
            random_state=RS,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        _skf = StratifiedKFold(5, shuffle=True, random_state=RS)

        _t1 = time.time()
        print("\n[ StratifiedKFold ] Rastgele Orman CV …")
        _oof_raw = cross_val_predict(_rf, _X, _y, cv=_skf, n_jobs=1, method="predict_proba")
        _rf.fit(_X, _y)
        _cls_order = list(_rf.classes_)
        _col_idx = np.array([_cls_order.index(c) for c in LABELS])
        _oof_ps = _oof_raw[:, _col_idx]
        _pred_oof = np.array(LABELS, dtype=object)[np.argmax(_oof_ps, axis=1)]
        print(
            "  accuracy=",
            round(accuracy_score(_y, _pred_oof), 4),
            " macro-F1=",
            round(f1_score(_y, _pred_oof, labels=LABELS, average="macro"), 4),
            " süre=",
            int(time.time() - _t1),
            "s",
        )

        _rows = [
            {
                "Model": "Rastgele Orman",
                "Protokol": "StratifiedKFold_5",
                "accuracy": accuracy_score(_y, _pred_oof),
                "macro_f1": f1_score(_y, _pred_oof, labels=LABELS, average="macro"),
                "weighted_f1": f1_score(_y, _pred_oof, labels=LABELS, average="weighted"),
                "macro_precision": float(
                    precision_recall_fscore_support(
                        _y, _pred_oof, labels=LABELS, average="macro", zero_division=0
                    )[0]
                ),
                "macro_recall": float(
                    precision_recall_fscore_support(
                        _y, _pred_oof, labels=LABELS, average="macro", zero_division=0
                    )[1]
                ),
            }
        ]
        _comp = pd.DataFrame(_rows)
        _comp.to_csv(os.path.join(MET, "01_model_karsilastirma.csv"), index=False)
        print("\n", _comp.to_string(index=False))

        _mets = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
        _mtit = ["Accuracy", "Makro F1", "Ağırlıklı F1", "Makro Precision", "Makro Recall"]
        _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
        _xr = np.arange(len(_mets))
        _v = [_comp[m].values[0] for m in _mets]
        _ax.bar(_xr, _v, 0.55, color="#3182bd")
        _ax.set_xticks(_xr)
        _ax.set_xticklabels(_mtit, rotation=15, ha="right")
        _ax.set_ylim(0, 1.05)
        _ax.set_title("Rastgele Orman — özet metrikler (StratifiedKFold OOF)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_07_rf_protokol_karsilastirma.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_07_rf_protokol_karsilastirma.png")

        _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
        _ax.bar(_xr, _v, 0.55, color="#6baed6")
        _ax.set_xticks(_xr)
        _ax.set_xticklabels(_mtit, rotation=15, ha="right")
        _ax.set_ylim(0, 1.05)
        _ax.set_title("Rastgele Orman — özet metrikler (aynı protokol)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_07b_rf_metrik_gruplu.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_07b_rf_metrik_gruplu.png")

        for _suffix, _yp in (("02_rf_rastgele_sinif_metrikleri", _pred_oof),):
            _cm = confusion_matrix(_y, _yp, labels=LABELS)
            _total = _cm.sum()
            _prows = []
            for _i, _lab in enumerate(LABELS):
                _tp = _cm[_i, _i]
                _fn = _cm[_i, :].sum() - _tp
                _fp = _cm[:, _i].sum() - _tp
                _tn = _total - _tp - _fn - _fp
                _pr = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
                _rec = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
                _sp = _tn / (_tn + _fp) if (_tn + _fp) else 0.0
                _f1 = 2 * _pr * _rec / (_pr + _rec) if (_pr + _rec) else 0.0
                _prows.append(
                    {
                        "Sinif": LABEL_TR[_lab],
                        "Precision": _pr,
                        "Duyarlilik (Recall)": _rec,
                        "Ozgulluk (Specificity)": _sp,
                        "F1": _f1,
                        "Destek": int(_tp + _fn),
                    }
                )
            _pcm = pd.DataFrame(_prows).set_index("Sinif")
            _wts = _pcm["Destek"].values
            _macro = _pcm[["Precision", "Duyarlilik (Recall)", "Ozgulluk (Specificity)", "F1"]].mean()
            _weighted = np.average(
                _pcm[["Precision", "Duyarlilik (Recall)", "Ozgulluk (Specificity)", "F1"]].values,
                axis=0,
                weights=_wts,
            )
            _pcm.loc["Makro ort."] = list(_macro) + [int(_wts.sum())]
            _pcm.loc["Agirlikli ort."] = list(_weighted) + [int(_wts.sum())]
            _pcm.round(4).to_csv(os.path.join(MET, _suffix + ".csv"), encoding="utf-8")
        print("\nSınıf metrikleri:\n", _pcm.round(3).to_string())

        _xtl = [LABEL_TR[x] for x in LABELS]
        _cm = confusion_matrix(_y, _pred_oof, labels=LABELS, normalize="true")
        _fig, _ax = plt.subplots(figsize=(6.8, 5.6))
        ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=_xtl).plot(
            ax=_ax, cmap="Blues", values_format=".2f"
        )
        _ax.set_title("Rastgele Orman — StratifiedKFold OOF")
        plt.setp(_ax.get_xticklabels(), rotation=25, ha="right")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_08_karmasiklik_rastgele.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_08_karmasiklik_rastgele.png")

        _cm_all = confusion_matrix(_y, _pred_oof, labels=LABELS)
        _f1_per = []
        for _i in range(len(LABELS)):
            _tp = _cm_all[_i, _i]
            _fn = _cm_all[_i, :].sum() - _tp
            _fp = _cm_all[:, _i].sum() - _tp
            _pr = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
            _rec = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
            _f1_per.append(2 * _pr * _rec / (_pr + _rec) if (_pr + _rec) else 0.0)

        _xi = np.arange(len(LABELS))
        _fig, _ax = plt.subplots(figsize=(9, 4.5))
        _ax.bar(_xi, _f1_per, 0.55, color="#3182bd")
        _ax.set_xticks(_xi)
        _ax.set_xticklabels(_xtl, rotation=20, ha="right")
        _ax.set_ylim(0, 1.05)
        _ax.set_ylabel("F1")
        _ax.set_title("Sınıf bazlı F1 (StratifiedKFold OOF)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_09_sinif_bazli_f1.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_09_sinif_bazli_f1.png")

        _pr_v, _re_v, _, _ = precision_recall_fscore_support(
            _y, _pred_oof, labels=LABELS, average=None, zero_division=0
        )
        _fig, _ax = plt.subplots(figsize=(9, 4.5))
        _xi2 = np.arange(len(LABELS))
        _w2 = 0.32
        _ax.bar(_xi2 - _w2 / 2, _pr_v, _w2, label="Precision", color="#9ecae1")
        _ax.bar(_xi2 + _w2 / 2, _re_v, _w2, label="Recall", color="#3182bd")
        _ax.set_xticks(_xi2)
        _ax.set_xticklabels(_xtl, rotation=20, ha="right")
        _ax.set_ylim(0, 1.05)
        _ax.legend()
        _ax.set_title("Sınıf başına Precision ve Recall (OOF)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_09b_sinif_precision_recall.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_09b_sinif_precision_recall.png")

        _dr = []
        for _v, _nm in ((0, "Engelsiz"), (1, "Engelli")):
            _m = _dis == _v
            _dr.append(
                {
                    "Grup": _nm,
                    "Ornek": int(_m.sum()),
                    "Accuracy": accuracy_score(_y[_m], _pred_oof[_m]),
                    "Makro F1": f1_score(_y[_m], _pred_oof[_m], labels=LABELS, average="macro"),
                }
            )
        pd.DataFrame(_dr).to_csv(os.path.join(MET, "04_engelli_kirilim.csv"), index=False)

        _per_u = (
            pd.DataFrame({"UserID": _g, "dis": _dis, "ok": _pred_oof == _y})
            .groupby("UserID")
            .agg(acc=("ok", "mean"), dis=("dis", "first"))
        )
        _per_u.to_csv(os.path.join(MET, "06_kisi_bazli_dogruluk.csv"))
        _fig, _ax = plt.subplots(figsize=(9, 4))
        _col2 = ["#fc8d59" if v == 1 else "#99d8c9" for v in _per_u["dis"]]
        _ax.bar(_per_u.index.astype(str), _per_u["acc"], color=_col2)
        _ax.set_title("Kişi bazlı doğruluk (OOF, örnek bazlı CV — makale ile aynı aile)")
        _ax.set_ylim(0, 1.05)
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_11_kisi_bazli_dogruluk.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_11_kisi_bazli_dogruluk.png")

        _fig, _ax = plt.subplots(figsize=(6, 4))
        _x2 = np.arange(2)
        _ax.bar(_x2 - _bw / 2, [_dr[0]["Accuracy"], _dr[0]["Makro F1"]], _bw, label="Engelsiz")
        _ax.bar(_x2 + _bw / 2, [_dr[1]["Accuracy"], _dr[1]["Makro F1"]], _bw, label="Engelli")
        _ax.set_xticks(_x2)
        _ax.set_xticklabels(["Accuracy", "Makro F1"])
        _ax.set_ylim(0, 1.05)
        _ax.legend()
        _ax.set_title("Engelli / engelsiz (StratifiedKFold OOF, RF)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_10_engelli_performans.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_10_engelli_performans.png")

        _imp = pd.Series(_rf.feature_importances_, index=IMU).sort_values(ascending=False)
        _imp.round(6).head(25).to_frame("onem").to_csv(os.path.join(MET, "05_oznitelik_onemi.csv"), encoding="utf-8")
        _fig, _ax = plt.subplots(figsize=(7.5, 6.2))
        _top = _imp.head(15).iloc[::-1]
        _ax.barh(_top.index, _top.values, color="#2c7fb8")
        _ax.set_title("Öznitelik önemi (IMU, tam veri RF)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_12_oznitelik_onemi.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_12_oznitelik_onemi.png")

        _rf30 = RandomForestClassifier(n_estimators=80, random_state=RS, n_jobs=-1, min_samples_leaf=2)
        _rf30.fit(_df[LOC + IMU].values, _y)
        _ia = pd.Series(_rf30.feature_importances_, index=LOC + IMU).sort_values().tail(15)
        _fig, _ax = plt.subplots(figsize=(7, 5))
        _clr = ["#d73027" if n in LOC else "#2c7fb8" for n in _ia.index]
        _ax.barh(_ia.index, _ia.values, color=_clr)
        _ax.set_title("30 özellik: konum sızıntısı (kırmızı=GPS)")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_13_konum_sizinti_onem.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_13_konum_sizinti_onem.png")

        _fig, _ax = plt.subplots(figsize=(14, 7))
        plot_tree(
            _rf.estimators_[0],
            feature_names=IMU,
            class_names=[LABEL_TR[c] for c in _rf.classes_],
            max_depth=3,
            filled=True,
            rounded=True,
            fontsize=8,
            impurity=False,
            ax=_ax,
        )
        _ax.set_title("Ormanın 1. ağacı (max_depth=3 gösterim)", fontsize=12)
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_14_rf_ornek_agac.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_14_rf_ornek_agac.png")

        _y_bin = label_binarize(_y, classes=LABELS)
        _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
        for _j, _lab in enumerate(LABELS):
            _fpr, _tpr, _ = roc_curve(_y_bin[:, _j], _oof_ps[:, _j])
            _ax.plot(_fpr, _tpr, lw=1.5, label=LABEL_TR[_lab])
        _fpr_m, _tpr_m, _ = roc_curve(_y_bin.ravel(), _oof_ps.ravel())
        _ax.plot(_fpr_m, _tpr_m, "k--", lw=2, label="Mikro-ortalama (OvR)")
        _ax.plot([0, 1], [0, 1], ":", color="gray")
        _ax.set_xlabel("False Positive Rate")
        _ax.set_ylabel("True Positive Rate")
        _ax.set_title("ROC (StratifiedKFold OOF olasılıkları, OvR)")
        _ax.legend(fontsize=7, loc="lower right")
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_15_rf_roc_ovr.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_15_rf_roc_ovr.png")

        _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
        for _j, _lab in enumerate(LABELS):
            _prec, _rec, _ = precision_recall_curve(_y_bin[:, _j], _oof_ps[:, _j])
            _ax.plot(_rec, _prec, lw=1.5, label=LABEL_TR[_lab])
        _ax.set_xlabel("Recall")
        _ax.set_ylabel("Precision")
        _ax.set_title("Precision–Recall (StratifiedKFold OOF)")
        _ax.legend(fontsize=7, loc="upper right")
        _ax.set_xlim(0, 1.05)
        _ax.set_ylim(0, 1.05)
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_16_rf_pr_egrileri.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_16_rf_pr_egrileri.png")

        _cm2 = confusion_matrix(_y, _pred_oof, labels=LABELS)
        _tot2 = _cm2.sum()
        _heat = []
        for _i in range(len(LABELS)):
            _tp = _cm2[_i, _i]
            _fn = _cm2[_i, :].sum() - _tp
            _fp = _cm2[:, _i].sum() - _tp
            _tn = _tot2 - _tp - _fn - _fp
            _pr = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
            _rec = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
            _sp = _tn / (_tn + _fp) if (_tn + _fp) else 0.0
            _f1 = 2 * _pr * _rec / (_pr + _rec) if (_pr + _rec) else 0.0
            _heat.append([_pr, _rec, _sp, _f1])
        _heat = np.array(_heat)
        _fig, _ax = plt.subplots(figsize=(6.5, 4.2))
        _im2 = _ax.imshow(_heat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        _ax.set_xticks(range(4))
        _ax.set_xticklabels(["Precision", "Recall", "Specificity", "F1"], rotation=15, ha="right")
        _ax.set_yticks(range(len(LABELS)))
        _ax.set_yticklabels(_xtl)
        _ax.set_title("Sınıf metrikleri ısı haritası (StratifiedKFold OOF)")
        for _ri in range(len(LABELS)):
            for _ci in range(4):
                _ax.text(_ci, _ri, f"{_heat[_ri, _ci]:.2f}", ha="center", va="center", fontsize=8, color="black")
        _fig.colorbar(_im2, ax=_ax)
        _fig.tight_layout()
        _fig.savefig(os.path.join(FIG, "fig_17_rf_sinif_metrik_isi.png"), bbox_inches="tight")
        plt.close(_fig)
        print("  şekil: fig_17_rf_sinif_metrik_isi.png")

        _summ = {
            "n_samples": int(len(_df)),
            "n_features_IMU": len(IMU),
            "n_estimators": N_TREES,
            "protocol": "StratifiedKFold_5_shuffle_RS42",
            "RF_oob_accuracy": float(_comp["accuracy"].iloc[0]),
            "RF_oob_macro_f1": float(_comp["macro_f1"].iloc[0]),
            "top5_features": list(_imp.head(5).index),
            "runtime_sec": round(time.time() - _t0, 1),
        }
        with open(os.path.join(MET, "07_ozet.json"), "w", encoding="utf-8") as _f:
            json.dump(_summ, _f, indent=2, ensure_ascii=False)
        print("\nÖzet:", json.dumps(_summ, indent=2, ensure_ascii=False))
