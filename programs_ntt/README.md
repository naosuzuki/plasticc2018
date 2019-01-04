CREST_auto
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

----

# 2018/7/13 更新
* 複数のタスクを同時に実行する方法についてメモ

   - CUDA MPS(Multi Process Service)を利用する
   - マルチGPU環境での利用に対応
   - ハイパーパラメータの探索で利用することを想定

# 2018/2/16 更新
* 追加のパッケージ  

        conda install h5py
        pip install hyperopt

* 回帰タスク(redshiftとsn_epoch)を学習、予測するためのソースコードを追加
* ハイパーパラメータの探索処理を追加
* クラス分類で利用できるモデルを通常のDNNのみに限定(ドメイン適応などの切り替えオプションを削除)
* クラス分類と2つの回帰の予測結果を一つのcsvにまとめる処理を追加

# 環境構築

## cudaとcudnnのインストール
必要なcudaとcudnnのバージョンはTensorflowのバージョンに依存しています。  
[Tensorflow/Releases](https://github.com/tensorflow/tensorflow/releases)で必要なバージョンを確認してください。

nvidiaの公式ページからそれぞれをダウンロードし、インストールします。

## Pythonのインストール
大まかな流れは次の通り  
1. Anacondaをインストール  
2. Anaconda内に仮想環境を構築  
3. 仮想環境の構築時に入らなかった残りのパッケージを追加

### Anacondaをインストール
Pythonは[Anaconda](https://www.anaconda.com/download/ "Downloads Anaconda")を使ってインストールします。

こちらから[Anaconda](https://www.anaconda.com/download/ "Downloads Anaconda")をダウンロードし、インストールを行います。  
ダウンロードするPythonのバージョンは2系と3系のどちらでも構いません。  

ダウンロードしたファイルを実行するとインストールが始まります。  
インストールディレクトリ、.bashrcへのパスの追加はインストーラに従ってデフォルトで構いません。  
(.bashrcがない場合はパスの追加は行われないはずです。)

### Anaconda内に仮想環境を構築
    conda create --name py3.5 python=3.5 anaconda

--nameの後の'py3.5'は構築する環境の名前です。わかりやすい、入力しやすい名前に変更してください。

### 残りのパッケージを追加
仮想環境の構築時に必要なパッケージの大半はインストールされていますが、Tensorflowなどを追加します。

    source activate py3.5  # 作成した仮想環境を有効化
    pip install tensorflow-gpu hyperopt
    conda install luigi tqdm xarray netCDF4 bottleneck h5py

### windows xgboost
http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/

# Real/Bogusの判定
## データの準備
データの配置先は任意です。  
ここでは、data/raw/param.lst/param.lst (param.lstというフォルダの中にparam.lstというファイルがある)にあるものとします。

src/dataにカレントディレクトリを移動して、以下のコマンドを実行します。

    source activate py3.5  # 作成した仮想環境を有効化
    python make_dataset_real_bogus.py RealBogusBandSplitter \
     --file-path=../../data/raw/param.lst/param.lst \
     --local-scheduler

オプションの説明
* --file-path:【省略可能】入力データのファイル。デフォルトは../../data/raw/param.lst/param.lst
* --working_dir:【省略可能】中間生成物の出力ディレクトリ。デフォルトは、../../data/interim/real_bogus
* --output_dir:【省略可能】バンドごとに分割されたバイナリファイルの出力ディレクトリ。デフォルトは、../../data/processed/real_bogus

コマンドを実行すると、 `output_dir` の中に以下の
* param_HSC-G.pickle
* param_HSC-I2.pickle
* param_HSC-R2.pickle
* param_HSC-Y.pickle
* param_HSC-Z.pickle

バンドごとに分割されたファイルが生成されます。
### メモ
* バンドの種類(名前)は、上記の5種類で固定です。変更がある場合はプログラムの修正が必要です。
* テスト(評価用)データも同じフォーマットで与えられるものとしています。9列目のreal/bogusもダミーの値でいいので、必須です。

## 学習
カレントディレクトリをsrc/modelsに移動して、以下のコマンドを実行します。

#### メモ
データを順位付けする関数(スコア関数)にDNNと混合ガウス分布の尤度比の2種類から選択できます。  
そのため、コマンドの形式が変更になりました。  
互換性のために古い形式のコマンドでスコア関数の選択(デフォルトでは今までと同じでDNNが選ばれる)できるようにはなっていますが、
無効なオプションを指定してもエラーにはならないので、やや扱いには注意が必要です。

### 学習(pauc_relaxed)　今までの形式
pAUCの緩和問題を計算します。  
緩和問題では負例についてスコアで並び替える必要がないので、通常のpAUCを計算するより高速な場合があります。

    python train_real_bogus_model.py pauc_relaxed \
     --data_path=../../data/processed/real_bogus/param_HSC-G.pickle \
     --epoch=100 \
     --regularizer_weights=1e-1 1e-1 \
     --output_dir=../../models/real_bogus/pauc/relaxed/HSC-G_test4 \

     --model_type=dnn \
     --min_epoch=500 \
     --increasing_ratio=2.0 \
     --improvement_threshold=1.0 \
     --positive_mixture=1 \
     --negative_mixture=1

オプションの説明
* --data_path:【必須】変換した入力データのパス。
* --output_dir:【必須】学習結果の出力先ディレクトリ。
* --regularizer_weights:【省略可能】realらしさの値(スコア)のネットワークの重みの正規化の強さ。デフォルトは0 0(数字が2個)。  
 ネットワークの構造に依存して個数が決まるので、おそらく将来のバージョンでは変更されます。  
 --model-type=dnnの時のみ有効です。その他の時は無視されます。
* --epoch:【省略可能】反復回数。デフォルトは1000
* --hidden_size:【省略可能】スコアネットワークの隠れ層の大きさ。デフォルトは100。  
 --model-type=dnnの時のみ有効です。その他の時は無視されます。
* --gamma:【省略可能】FPR(false positive rate)が小さい領域のデータを重視する度合い。0<gamma<1  
 小さい方がより重視する。
* --seed:【省略可能】入力データを訓練、バリデーション、テストに分割するときのシード値。デフォルトは0
* --split_ratio:【省略可能】入力データを訓練、バリデーション、テストに分割する割合。デフォルトは0.8 0.1 0.1
* --resume:【省略可能】学習を再開するときに指定する。

* --model_type:【省略可能】dnn, gmm-full, gmm-diagonalのいずれかを指定。省略した場合はdnn  
  - dnn: 今までと同じ。スコア関数にニューラルネットを利用します。
  - gmm-full: スコア関数に混合正規分布の対数尤度比を利用します。正規分布の共分散行列はフルランク行列。
  - gmm-diagonal: スコア関数に混合正規分布の対数尤度比を利用します。正規分布の共分散行列は対角行列。
* --min_epoch:【省略可能】反復回数の下限値。デフォルトは500
* --increasing_ratio:【省略可能】バリデーションスコアが大きく改善した時に反復回数を延長する割合。  

        max(現在の反復回数×increasing_ratio, min_epoch)

 が新たな反復回数の下限値となる。
* --improvement_threshold:【省略可能】バリデーションスコアが大きく改善したと判定するときの閾値。  

        現在のバリデーションスコア×improvement_threshold > 今までのバリデーションスコアの最良値

 を満たす時に大きく改善したとする。
* --positive_mixture:【省略可能】正例らしさ(尤度)を表現するGMMの混合数。  
 --model-typeがgmm-fullもしくはgmm-diagonalの場合のみ有効。
* --negative_mixture:【省略可能】負例らしさ(尤度)を表現するGMMの混合数。  
 --model-typeがgmm-fullもしくはgmm-diagonalの場合のみ有効。

### 学習(pauc_exact)　今までの形式
通常のpAUCを計算します。  
pAUCを計算するFPR(False Positive Rate)の値betaが小さい場合は、こちらの方が計算が高速な場合が多いです。

    python train_real_bogus_model.py pauc_exact \
     --data_path=../../data/processed/real_bogus/param_HSC-G.pickle \
     --epoch=100 \
     --regularizer_weights=1e-1 1e-1 \
     --output_dir=../../models/real_bogus/pauc/relaxed/HSC-G_test4 \

     --model_type=dnn \
     --min_epoch=500 \
     --increasing_ratio=2.0 \
     --improvement_threshold=1.0 \
     --positive_mixture=1 \
     --negative_mixture=1

オプションはpauc_relaxedの場合とほとんど共通です。  
pauc_relaxedに固有なパラメータgammaの代わりにpauc_exactに固有なパラメータbetaが追加されます。

* --beta:【省略可能】FPRが\[0, beta\]の範囲でpAUCを計算します。デフォルトは0.1

### 学習(pauc)　新しい形式
新形式のコマンドです。  
こちらでは、上記のpauc_relaxedとpauc_exactと等価な処理を行えます。  
スコア関数にDNNを利用する時に `positive-mixture` などを指定するとエラーになるので、こちらの方が安全です。

コマンドは以下のようになります。

    python train_real_bogus_model.py pauc (relaxed か exact) (dnn か gmm) (gmmの場合のみfull か diagonal) [option]

括弧の部分は二つの中から一つを選択するという意味です。

オプションはpauc_relaxedやpauc_exactとほぼ共通ですが、
区切りの記号を_(アンダースコア)から-(ダッシュ)に変更しています。

#### 全体で共通のオプション
* --data-path:【必須】
* --output-dir:【必須】
* --seed:【省略可能】
* --split-ratio:【省略可能】
* --max-epoch:【省略可能】今までのepochに相当
* --min-epoch:【省略可能】
* --resume:【省略可能】
* --batch-size:【省略可能】
* --validation-frequency:【省略可能】
* --increasing-ratio:【省略可能】
* --improvement-threshold:【省略可能】

#### relaxedで指定できるオプション
* --gamma:【省略可能】

#### exactで指定できるオプション
* --beta:【省略可能】

#### dnnで指定できるオプション
* --hidden-size:【省略可能】
* --regularizer-weights:【省略可能】

#### gmmで指定できるオプション
* --positive-mixture:【省略可能】
* --negative-mixture:【省略可能】

#### 既知の問題、仕様上の制限
* 学習結果のファイルにはDNNでのスコアネットワークの構造の情報は含まれていないので、将来のバージョンでスコアネットワークの部分のプログラムを変更した場合は、正しいネットワークを構築できないので、学習の再開、評価ができなくなります。  
 できるだけ早い段階で、ネットワークの構造も保存するように変更を予定。

### 学習(random_forest)
    python train_real_bogus_model.py random_forest \
     --data_path=../../data/processed/real_bogus/param_HSC-G.pickle \
     --output_dir=../../models/real_bogus/random_forest/random_forest/HSC-G_test1 \
     --n_jobs=-1 \
     --n_estimators=100

オプションの説明
* --data_path:【必須】
* --output_dir:【必須】
* --n_estimators:【省略可能】決定木の本数。
* --n_jobs:【省略可能】並列スレッド数。-1は全てのスレッドを利用する。
* --seed:【省略可能】
* --split_ratio:【省略可能】
* --resume:【省略可能】

### 学習(xgboost)

    python train_real_bogus_model.py xgboost \
     --data_path=../../data/processed/real_bogus/param_HSC-G.pickle \
     --output_dir=../../models/real_bogus/random_forest/random_forest/HSC-G_test1 \
     --n_jobs=-1 \
     --n_estimators=100

## 予測(pauc)

    python predict_real_bogus_model.py pauc \
     --data-path=../../data/processed/real_bogus/param_HSC-G.pickle \
     --output-dir=../../models/real_bogus/pauc/relaxed/HSC-G_test4

オプションの説明
* --data-path:【必須】変換した入力データのパス。
* --output-dir:【必須】学習結果の出力先ディレクトリ。

`output-dir` にdata-all.csvが作られます。  
また、同時にdata-train.csv, data-validation.csv, data-test.csvも作られます。

data-all.csvは入力データからそのまま予測した結果です。  
data-all.csvの1行目は、(バイナリに変換前の)入力データの1行目の値から予測したものに対応します。  
1列目はrealらしさの値(スコア)、2列目は入力データのreal/bogusそのままの値です。

#### 注意
入力データのバンドと学習器が学習したバンドが同じであるかをチェックする処理はありません。  
そのため、入力データと学習器の正しい組み合わせを指定してください。

## 予測(random_forest)
予測(pauc_relaxed)と同じです。  
`pauc_relaxed` の代わりに `random_forest` を指定してください。

    python predict_real_bogus_model.py random_forest \
     --data_path=../../data/processed/real_bogus/param_HSC-G.pickle \
     --output_dir=../../models/real_bogus/random_forest/random_forest/HSC-G_test1

# 超新星タイプの判定
## データの準備

### 学習データ
`src/data` を相対パスの起点として以下の操作を説明します。  
また、 `source activate py3.5` で仮想環境が有効になっているものとします。

学習データを任意のディレクトリに配置します。  
以下の実行例では `../../data/raw/dataset_all` にSimSN_dataset_171128_1_Ia.tar.gz, SimSN_dataset_171128_1_nonIa.tar.gzを解凍してできたフォルダを配置しています。

以下の例の通りに実行すると `../../data/processed/dataset_all/train` に
`dataset.tr-6classes.nc` , `dataset.va-6classes.nc` , `dataset.te-6classes.nc` が作られます。

    # 実行例：訓練データ
    python make_dataset.py TrainData \
    --train-root=../../data/raw/dataset_all \
    --sn-types=\{\"Ia\":0,\"Ib\":1,\"Ibc\":1,\"Ic\":2,\"IIL\":3,\"IIN\":4,\"IIP\":5\} \
    --n-workers=7 \
    --working-dir=../../data/interim/dataset_all/train \
    --suffix=6classes \
    --output-dir=../../data/processed/dataset_all/train \
    --local-scheduler

オプションの説明
* --train-root: 【必須】学習データが配置されているディレクトリ。再帰的にすべてのディレクトリが探索されるので、ディレクトリ構成は自由です。
* --sn-types: 【必須】学習データのタイプと分類問題のクラスラベルの対応付けをします。  
 ここで `{\"Ia\":0,\"Ib\":1,\"Ibc\":1,\"Ic\":1,\"IIL\":1,\"IIN\":1,\"IIP\":1}` とするとIaにラベル0、それ以外にラベル1を割り振ります。  
 また、 `{\"Ia\":0,\"IIP\":1}` とすると学習用のバイナリデータが、IaとIIPのみで作成されます。  
 bashなどでは{と}の前にバックスラッシュが必要。
* --working-dir: 【必須】中間生成物の出力ディレクトリ。変換処理の各工程が完了しているかの判定は中間生成物の有無で判定されます。そのため、何らかの理由で学習データを変更した場合は、その超新星のタイプの中間生成物を削除する必要があります。
* --output-dir: 【必須】バイナリに変換した学習用データの保存先ディレクトリ。
* --suffix: 【省略可能】同じ学習データから2クラス分類用のバイナリと6クラス分類用のバイナリを作成するときにそれぞれのデータを区別するために利用する。そのような使い方を想定。
* --n-workers: 【省略可能】individual_parameter.datを読み込む処理ののスレッド数。デフォルト値は1。
* --split-ratio: 【省略可能】学習データを訓練・バリデーション・テストに分割するときの比率。デフォルトでは8:1:1で、それぞれは訓練・バリデーション・テストの比率。
 `--split-ratio=(1.0,2.0,3.0)` のように括弧の中に値を記述します。  
 bashなどでは(と)の前にバックスラッシュが必要
* --seed: 【省略可能】データを分割する際の乱数のシード値。デフォルトは0x5eed。

### 実際の観測データ
実際の観測データをバイナリ形式に変換します。コマンドは学習データと似ています。

    python make_dataset.py TestData \
    --test-root=C:\Users\imoto\Documents\CREST3\data\observation\test\170519 \
    --sn-types={\"Ia\":0,\"Ib\":1,\"Ibc\":1,\"Ic\":2,\"IIL\":3,\"IIN\":4,\"IIP\":5} \
    --n-workers=7 \
    --working-dir=../../data/interim/dataset_all/test \
    --suffix=6classes \
    --output-dir=../../data/processed/dataset_all/test \
    --label-file-path=../../external/label_list.dat \
    --local-scheduler

オプションの説明
* --test-root: 【必須】観測データが配置されているディレクトリ。
* --working-dir: 【必須】中間生成物が配置されるディレクトリ。
* --output-dir: 【必須】バイナリに変換した観測データが配置されるディレクトリ。
* --sn-types: 【省略可能】精度を評価するときにデータの識別結果と共に正解ラベルを出力するために利用します。指定する場合は、学習データを作成した時に与えたものを同じ値を指定すべきと思います。
* --label-file-path: 【省略可能】観測データの名前と超新星のタイプを関連付けるためのファイル。観測データの名前と超新星のタイプの組が1行に一つずつ並んでいるものとする。  
 また、この時--sn-typesを指定する必要がある。
* --n-workers: 【省略可能】学習データの作成のものと同じ
* --suffix: 【省略可能】学習データの作成のものと同じ


### 既知の問題
* (データの準備の工程で)処理がいつまでも終わらない上に、CPUの利用率も低いままである。  
 netcdf形式のファイル(拡張子: .nc)の読み込みでエラーになっている可能性が高いです。  
 ファイルを保存する段階でオプションの指定が必要になるので、ソースコードの修正が必要になります。  
 修正のためにターミナルの出力を添えてメールでお知らせください。
* 終了時にエラーが出る  
 以下のような感じ

        Exception ignored in: <function WeakValueDictionary.\_\_init__.<locals>.remove at 0x0000000007D74B70>
        Traceback (most recent call last):
          File "C:\Users\imoto\Anaconda3\envs\py3.5\lib\weakref.py", line 117, in remove
        TypeError: 'NoneType' object is not callable

 処理自体は正常に終了しているので問題はありません。  
 Tensorflowのメモリの解放時にエラーが出ているようですが、ライブラリの内部のことなので、対策はありません。

## ハイパーパラメータの探索
探索対象のハイパーパラメータは以下の4つです。

1. DNNの隠れ層の大きさ
2. DNNのdropoutの確率
3. 入力値を0にする確率
   * fluxがNaNの場合の再現のため
4. magnitudeにさらにノイズを加える確率
   * 外れ値の再現のため、ノイズの分布は標準正規分布で固定
   * 学習時は常にfluxにノイズを加えたものをmagnitudeに変換

ハイパーパラメータの探索にはhyperoptを利用します。  
探索はシングルプロセスか(分散)マルチプロセスで行えます。  
マルチプロセスの場合は1つのタスクに対して、3プロセスまで利用できるようです。  
これは、hyperoptの仕様によるものだと思われます。

### シングルプロセスの場合
実行例

    python train_model.py sn_class search \
    --epoch=100 \
    --band_data=\{\"i\":3,\"z\":3\} \
    --method=modified \
    --train_data_path=../../data/processed/dataset.tr-2classes.nc \
    --validation_data_path=../../data/processed/dataset.va-2classes.nc \
    -test_data_path=../../data/processed/dataset.te-2classes.nc \
    --output_dir=../../models/outputs \
    --n_iterations=100

上の例を実行するとoutput_dirで指定したディレクトリに `best_parameter.json` が
生成されます。  
`best_parameter.json` は探索の結果や検証データに対する精度を含みます。

実行は例のように `python train_model.py <タスクの種類> search` の後に
オプションが続きます。  
タスクの種類の種類は
* sn_class: 超新星のタイプについてのクラス分類
* redshift: redshiftの値についての回帰
* sn_epoch: 明るさのピークの日数についての回帰

の3種類です。  
オプションは
* --epoch: 各ハイパーパラメータの探索点で、DNNの学習のエポック数
* --batch_size:  学習にミニバッチで読み込むデータ数
* --band_data: DNNの入力に含めるバンドの種類とその観測数  
  バンドとその数を例えば、 `\{\"i\":3,\"z\":3\}` の形で与える
* --method: fluxをmagnitudeに変換する方法  
  指定できる値は次の二つです
  * traditional: log10で変換
  * modified: arcsinhで変換
* --train_data_path: バイナリファイルに変換した学習データのパス
* --validation_data_path: バイナリファイルに変換した検証データのパス
* --test_data_path: バイナリファイルに変換したテストデータのパス
* --output_dir: ハイパーパラメータの探索結果を出力するディレクトリ
* --n_iterations: ハイパーパラメータの探索回数
* --use_redshift: 入力にredshiftの値を含めるかどうか  
  タスクの種類がsn_classかsn_epochの場合のみ指定可能
* --output_size: 超新星のタイプの個数  
  タスクの種類がsn_classの場合のみ指定可能

### マルチプロセスの場合
実行例

    # mongoデータベースを生成する適当なディレクトリに移動
    # データベースのサービスを起動
    mongod --dbpath . --port 1234

    # 別のターミナルから実行
    python train_model.py sn_class search_parallel \
    --epoch=100 \
    --band_data=\{\"i\":3,\"z\":3\} \
    --method=modified \
    --train_data_path=../../data/processed/dataset.tr-2classes.nc \
    --validation_data_path=../../data/processed/dataset.va-2classes.nc \
    -test_data_path=../../data/processed/dataset.te-2classes.nc \
    --output_dir=../../models/outputs \
    --n_iterations=100 \
    --hostname=localhost \
    --port=1234 \
    --db_name=ex20180216

    # さらに別のターミナルから実行
    export PYTHONPATH=/path/to/installed/dir/crest_auto/src/models:$PYTHONPATH
    hyperopt-mongo-worker --mongo=localhost:1234/ex20180216 --poll-interval=10

並列でのハイパーパラメータの探索はhyperoptの機能を利用しています。  

上記の実行例では、データベースのサービス、探索のホストプロセス、ワーカプロセスが
それぞれ同じ計算機上で実行されていますが、異なる計算機(分散環境)で実行することも可能です。

実行はシングルプロセスと同様に `python train_model.py <タスクの種類> search_parallel` の後にオプションが続きます。
オプションはシングルプロセスの場合のものに以下の三つが追加されています。
* hostname: mongodを起動した計算機の名前
* port: mongodを起動するときに指定したポート番号
* db_name: ハイパーパラメータの探索結果を区別するための名前  
  ただし、さらに内部でデータセットの厳選の有無やmagnitudeへの変換方法などに依存した
  `exp_key` でそれぞれの結果を区別しています。  
  そのため、このオプションでは、日付などでどのデータセットを区別できれば十分です。

#### 探索結果を区別する方法について
いくつもの設定を同時に実行し、かつそれらの探索結果を区別する方法はいくつかあります。

1. 設定ごとにmongodを起動する。このとき、dbpathやportが重複しないようにする。
2. mongodのプロセスは共通で、db_nameをそれぞれ別のものにする。
3. mongodのプロセスとdb_nameは共通でexp_keyをそれぞれ別のものにする。

今回は3番の方法を利用しています。  
hyperopt-mongo-workerがexp_keyに対しては固定化されないので、
実験設定ごとにワーカを用意する必要がなく、複数の実験設定でワーカを共有できます。  
ワーカ数が上限に満たない場合でも効率的に計算できます。

### 複数のタスクを同時に実行する方法 (2GPU環境)
基本的には、一つのタスクにつき一つのGPUが必要です。
GPUの利用率が低い場合は、そのまま単純に複数のタスクを実行できますが、
各タスクがそれぞれ排他的に実行されるので、効率は期待できません。

そこで、MPS(Multi Process Service)と呼ばれるものを利用します。
ソースコードはそのままで呼び出す際にいくつかコマンドを実行することで、複数のタスクを同時に実行できます。

実行の手順は
1. MPSを起動
2. タスクを実行
3. MPSを停止
です。

#### 1. MPSを起動
以下のような内容のシェルスクリプトを実行します。

```start_mps.sh
#!/usr/bin/env bash
N_GPUS=2
for ((i=0;i<${N_GPUS};++i)) do
    mkdir /tmp/mps_$i
    mkdir /tmp/mps_log_$i
    export CUDA_VISIBLE_DEVICES=$i
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$i
    nvidia-cuda-mps-control -d
done
```

2種類のディレクトリのパスは任意ですが、それぞれのディレクトリは異なっている必要があります。
また、ディレクトリが予め存在しないことを前提としています。

#### 2. タスクを実行
以下のように通常のタスクを実行するコマンドの前にCUDA_MPS_PIPE_DIRECTORYの指定を追加します。
/tmp/mps_0, /tmp/mps_1はstart_mps.shで作成したディレクトリです。
(/tmp/mps_$iの$iがfor文で0と1に置き換わっています)

```
CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0 python train_model.py (略)
CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_1 python train_model.py (略)
```

`CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0`を指定するとGPU0、`CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_1`を指定するとGPU1で実行されます。
2個のGPUへのタスクの割り当てはGPUの利用率を見ながら行うことになります。

#### 3. MPSを停止
以下のような内容のシェルスクリプトを実行します。

```stop_mps.sh
#!/usr/bin/env bash
N_GPUS=2
for ((i=0;i<${N_GPUS};++i)) do
    export CUDA_MPS_PIPE_DIRECTORY=tmp/mps_$i
    export CUDA_VISIBLE_DEVICES=$i
    echo quit | nvidia-cuda-mps-control
    rm -rf /tmp/mps_$i
    rm -rf /tmp/mps_log_$i
done
```

削除しているディレクトリは1.MPSの起動で作成したディレクトリと同じです。
ここでディレクトリを削除しているので、次回のMPSの起動時には/tmp/mps_0などのディレクトリを作成できます。

## 学習
タスクの種類ごとにDNNの学習を行います。

実行例

    python train_model.py sn_class optimize \
    --epoch=100 \
    --band_data=\{\"i\":3,\"z\":3\} \
    --method=modified \
    --train_data_path=../../data/processed/dataset.tr-2classes.nc \
    --validation_data_path=../../data/processed/dataset.va-2classes.nc \
    -test_data_path=../../data/processed/dataset.te-2classes.nc \
    --output_dir=../../models/sn_class/outputs
    --parameter_path=../../models/outpus/best_parameter.json

オプションはほぼハイパーパラメータの探索の時と同じです。
`python train_model.py <タスクの種類> optimize` の後にオプションが続きます。
探索対象だったハイパーパラメータの値を指定するためのオプションが追加されています。
* --parameter_path: 探索結果のファイルから値を読み取ります。
* --hidden_size: DNNの隠れ層の大きさ。parameter_pathと同時にしてされた場合は、こちらの値が優先される。
* --dropout_rate: DNNのdropoutの確率。parameter_pathと同時にしてされた場合は、こちらの値が優先される。
* --outlier_rate: 学習データのmagnitudeにノイズを加える確率。parameter_pathと同時にしてされた場合は、こちらの値が優先される。
* --blackout_rate: 学習データのmagnitudeを0にする確率。parameter_pathと同時にしてされた場合は、こちらの値が優先される。

学習を行うと `output_dir` に学習済みのモデル `model.h5` と学習結果をまとめた `summary.json` が生成されます。  
`output_dir` は任意のディレクトリを指定できますが、出力されるファイル名が固定なので、タスクの種類ごとに別のディレクトリを指定してください。

## 予測
学習済みのモデルを用いて各データの識別結果をcsv形式で出力します。

以下に実行例を示します。

    python predict_mode.py \
    sn_class \
    --train_data_path=../../data/processed/dataset.tr-2classes.nc \
    --validation_data_path=../../data/processed/dataset.va-2classes.nc \
    -test_data_path=../../data/processed/dataset.te-2classes.nc \
    --real_data_path=../../data/processed/dataset.test.all-2classes.nc \
    --output_dir=../../models/sn_class/outputs

各タスクのモデルの予測は `python predict_model.py <タスクの種類>` という形式になります。  
オプションの説明  
* --output_dir: 訓練時に指定した値をセットしてください。  
 このディレクトリの中にあるモデルを読み込みます。また、csv形式の識別結果が出力されます。
* --train_data_path: バイナリファイルに変換した学習データのパス
* --validation_data_path: バイナリファイルに変換した検証データのパス
* --test_data_path: バイナリファイルに変換したテストデータのパス
* --real_data_path: 実際の観測データのパス。
* --batch_size: データを評価するときのバッチサイズ。  
 学習時の設定と異なってかまいません。バッチサイズが大きい方がデータ全体の評価にかかる時間が短くなります。

予測結果(6クラス分類)のcsvファイルの冒頭を示します。

    ,0,1,2,3,4,5,label
    dstt,0.451279878616333,0.5487168431282043,5.561190619118861e-07,1.4001553836351377e-06,7.100075549715257e-07,5.928444011260581e-07,-1.0
    dstx,0.6890401244163513,0.3109596073627472,8.634955150910173e-08,6.882889636017353e-08,1.1020971157904569e-07,3.865379838430272e-08,-1.0
    dsui,0.8880928754806519,0.11190711706876755,4.0022694360042355e-12,7.764042013469474e-12,2.713127032066831e-12,1.408779343181621e-12,-1.0


1行目は各列の内容を示すヘッダです。2行目以降が判別結果となっています。  
各列は左から順に天体の名前、0番のクラスに対応する出力、1番のクラスに対応する出力、…、最後は正解のクラスラベルです。  
実際の観測データの正解ラベルは、データをバイナリ形式に変換する際に `--label-file-path` で指定されたファイル内に記述されていればここで表示されます。正解のラベルが与えられていない場合は代わりに無効な値として `-1` が表示されます。

### 予測結果の統合
タスクの種類ごとに別々のファイルに予測結果が出力されるので、それらを一つのファイルに統合します。

    python merge_results.py \
    --sn_class_dir=../../models/180206/sn_class/outputs \
    --redshift_dir=../../models/180206/redshift/outputs \
    --sn_epoch_dir=../../models/180206/sn_epoch/outputs \
    --output_dir=../../models/180206/merged/outputs

`output_dir` にreal.csvというファイルができます。  
実行結果の冒頭を例として示します。

    ,0,1,2,3,4,5,redshift,sn_epoch
    dtyl,0.9140303730964661,0.08596957474946976,3.3899110876020274e-16,1.1902935785246976e-15,1.139213388989425e-16,8.551475228889965e-17,0.6206176280975342,37.75680160522461
    dtvz,0.9119950532913208,0.08800499141216278,2.4868902220637284e-17,9.961165031481206e-17,5.229595478418464e-18,4.290701642598312e-18,0.519873857498169,38.440364837646484
    dtwl,0.9103845357894897,0.08961549401283264,1.6937042097857619e-18,3.4589740182180803e-18,2.340886236842976e-19,4.455832742258039e-19,0.5835894346237183,36.801143646240234

各列は左から順にデータの名前、クラス判別の各クラスの予測結果(確率)、redshiftの予測値、sn_epochの予測値となっています。  
データはクラスの確率が降順になる様に並び替えています。  
並び替えの優先順位は0番のクラスが最も高く、N-1番のクラス(Nクラス分類)が最も低くなっています。
