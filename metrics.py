import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns
import pandas as pd


if __name__ == '__main__':
    tree_types = ['Alder', 'Beech', 'Birch', 'Chestnut', 'Ginkgo biloba', 'Hornbeam', 'Horse chestnut',
                  'Linden', 'Oak', 'Oriental plane', 'Pine', 'Spruce']
    tree_types_ru = ['Ольха', 'Бук', 'Береза', 'Каштан', 'Гинкго билоба', 'Граб', 'Конский каштан',
                     'Липа', 'Дуб', 'Платан восточный', 'Сосна', 'Ель']

    con_mat =[[7, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 1, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 7, 0],
              [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 8]]
    con_mat = np.array(con_mat)
    # con_mat = tf.math.confusion_matrix(generator.labels, y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=tree_types_ru,
                              columns=tree_types_ru)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=cm.Blues)
    plt.tight_layout()
    plt.ylabel('Действительные')
    plt.xlabel('Предсказанные')
    plt.show()