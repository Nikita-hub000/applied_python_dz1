import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    data = pd.read_csv("output.csv")
    data = data.drop('CLIENT_ID', axis=1)
    data = data.drop('AGREEMENT_RK', axis=1)
    return data


def main():
    st.title("EDA с Streamlit")

    data = load_data()

    st.subheader("Превью данных")
    st.write(data.head())

    st.subheader("Графики распределений признаков")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        st.write(f"### {column} распределение")
        fig, ax = plt.subplots()
        ax.hist(data[column], bins=30)
        st.pyplot(fig)

    st.subheader("Матрица корреляций")
    corr_matrix = data.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    # plt.figure(figsize=(12, 8))

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Графики зависимостей целевой переменной и признаков")
    numeric_columns_without_TARGET = numeric_columns.drop('TARGET')
    for column in numeric_columns_without_TARGET:
        st.write(f"### {column} относительно TARGET")
        fig, ax = plt.subplots()
        if column == 'PERSONAL_INCOME' or column == 'AGE':
            sns.lineplot(x=column, y=data['TARGET'], data=data, ax=ax)
        else:
            sns.barplot(x=column, y=data['TARGET'], data=data, ax=ax)
        st.pyplot(fig)

    st.subheader("Числовые характеристики распределения числовых столбцов")
    st.write(data.describe())


if __name__ == "__main__":
    main()
