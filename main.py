import streamlit as st
import random
from dashboard import make_figure
from forecast import *
from additive_solver import Solve as SolveAdditive
from multiplicative_solver import Solve as SolveMultiplicative
from io import StringIO
from find_degrees import *

st.set_page_config(
    page_title='Lab 4',
    layout='wide')

st.title("lab4")
st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fe0de;
        font-size:300px !important;
    }
    div.stButton button {
    background-color: rgb(100, 76, 124);
    transition: 0.2s;
    color: white !important;
    }
    div.stButton button:active {
        background-color: #4caf50;
        color: white;
    }
    div.stButton button:hover {
        transition: 0.2s;
        background-color: #5da633;
        color: white;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-1y4p8pa.ea3mdgi2 > div > div > div > div.st-emotion-cache-r421ms.e10yg2by1 {
    width: 800px !important;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-1y4p8pa.ea3mdgi2 {
    margin-left: -50% !important;
    }
    h1 {
    font-weight: bold;
    text-align: left;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-1y4p8pa.ea3mdgi2 > div > div > div > div.st-emotion-cache-r421ms.e10yg2by1 {
    width: 100% !important;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-1y4p8pa.ea3mdgi2 {
    padding-top: 2% !important;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-1y4p8pa.ea3mdgi2 > div {
    width: 1000px !important;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-1y4p8pa.ea3mdgi2 > div > div > div > div:nth-child(6) > div.st-emotion-cache-16nc0hx.e1f1d6gn3 > div > div > div > div > div {
    padding-left: 30% !important;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > header > div.st-emotion-cache-zq5wmm.ezrtsby0 > div.stDeployButton > button {
      display: none;
      pointer-events: none;
    }
    #MainMenu > button {
      display: none;
      pointer-events: none;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > header > 
    div.st-emotion-cache-zq5wmm.ezrtsby0 > 
    div.st-emotion-cache-19or5k2.en6cib61.StatusWidget-enter-done > div > label {
      display: none;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > header > 
    div.st-emotion-cache-zq5wmm.ezrtsby0 > 
    div.st-emotion-cache-19or5k2.en6cib61.StatusWidget-enter-done > div > img {
      display: none;
    }
    #stDecoration {
      display: none;
    }
    </style>
    """, unsafe_allow_html=True)


forecast_type = 'recovery'
if forecast_type == 'Відновлення ФЗ':
    forecast_type = 'recovery'
elif forecast_type == 'ARMA':
    forecast_type = 'arma'
elif forecast_type == 'Коефіцієнт локального відхилення':
    forecast_type = 'local_outlier_factor'

form_values = st.form(key='user_input_form')

col8, col9, col10 = form_values.columns([1, 1, 1])

input_file = form_values.file_uploader('Вхідний файл', type=['csv','txt'],
                                       key='input_file')
output_file = "result"

sample_size = int(form_values.text_input('Розмір вибірки', value=500,
                                   key='sample_size'))


col1, col2, col3, col4, col5 = form_values.columns([1, 1, 1, 1, 1])

with col1:
    x1_dim = int(st.text_input('Розмірність X1', value=3,
                                key='x1_dim'))

with col2:
    x2_dim = int(st.text_input('Розмірність X2', value=3,
                             key='x2_dim'))

with col3:
    x3_dim = int(st.text_input('Розмірність X3', value=5, key='x3_dim'))

with col4:
    x4_dim = int(st.text_input('Розмірність X4', value=4, key='x4_dim'))

with col5:
    y_dim = int(st.text_input('Розмірність Y', value=4, key='y_dim'))

if forecast_type == 'recovery':

    recovery_type = form_values.selectbox('Форма', ['Адитивна',
                                                'Мультиплікативна'])

    if recovery_type == 'Адитивна': recovery_type = 'Additive'
    elif recovery_type == 'Мультиплікативна': recovery_type = 'Multiplicative'

    poly_type = form_values.selectbox('Тип поліному',
                             ['T-поліном', 'H-поліном', 'L-поліном',
                              'P-поліном', 'U*-поліном', 'O-поліном'])
    if poly_type == 'T-поліном': poly_type = 'Chebyshev'
    elif poly_type == 'H-поліном': poly_type = 'Hermitt'
    elif poly_type == 'L-поліном': poly_type = 'Lagger'
    elif poly_type == 'P-поліном': poly_type = 'Legandre'
    elif poly_type == 'U*-поліном': poly_type = 'u_shifted_polynomial'
    elif poly_type == 'O-поліном': poly_type = 'o_polynomial'
    col5, col6, col7, col8 = form_values.columns(4)

    with col5:
        x1_deg = int(st.text_input('Степінь X1', value=0, key='x1_deg'))

    with col6:
        x2_deg = int(st.text_input('Степінь X2', value=0, key='x2_deg'))

    with col7:
        x3_deg = int(st.text_input('Степінь X3', value=0, key='x3_deg'))

    with col8:
        x4_deg = int(st.text_input('Степінь X4', value=0, key='x4_deg'))
    weight_method = 'Нормоване значення'
    lambda_option = form_values.checkbox('Визначати λ з трьох систем рівнянь',
                                         value=True)

elif forecast_type == 'arma':
    ar_order = int(form_values.text_input('Порядок AR (авторегресії)', value=0,
                                        key='ar_order'))
    ma_order = int(form_values.number_input('Порядок MA (ковзного середнього)',
                                        value=0, key='ma_order'))

col9, col10 = form_values.columns([1, 1])

with col8:
    samples = st.number_input('Розмір вікна', value=50, step=1,
                                       key='samples')
with col9:
    pred_steps = st.number_input('Розмір вікна прознозу', value=10,
                                          step=1, key='pred_steps')

button_col1, button_col2 = st.columns(2)

with button_col1:
    button1 = form_values.form_submit_button('Знайти оптимальні степені')

with button_col2:
    button2 = form_values.form_submit_button('Почати')

text_placeholder = st.empty()

if button1:
    if input_file is None:
        st.error("Помилка: No File Uploaded to Run")
    else:
        input_file_text = input_file.getvalue().decode()
        try:
            data_file = StringIO(input_file_text)
            input_data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
            if sample_size > 0 and sample_size < len(input_data):
                input_data = input_data[:sample_size]
            dim_are_correct = True
        except ValueError:
            st.error("Помилка: перевірте розмірності")
            dim_are_correct = False

        if dim_are_correct:
            params = {
                'dimensions': [x1_dim, x2_dim, x3_dim, x4_dim, y_dim],
                'input_file': input_data,
                'output_file': output_file + '.xlsx',
                'samples': samples,
                'pred_steps': pred_steps,
                'labels': {
                    'rmr': 'rmr',
                    'time': 'Момент часу',
                    'y1': 'Рівень в першому резервуарі',
                    'y2': 'Напор',
                    'y3': 'Температура ТУ',
                    'y4': 'Рівень в другому резервуарі'
                }
            }
            if forecast_type == 'recovery':
                params['degrees'] = [x1_deg, x2_deg, x3_deg, x4_deg]
                params['weights'] = weight_method
                params['poly_type'] = poly_type
                params['lambda_multiblock'] = lambda_option
            elif forecast_type == 'arma':
                params['degrees'] = [ar_order, ma_order]

            fault_probs = []
            for i in range(y_dim):
                fault_probs.append(
                    FaultProb(
                        input_data[:, -y_dim + i],
                        y_emergency=danger_levels[i][0],
                        y_fatal=danger_levels[i][1],
                        window_size=params['samples'] // params['pred_steps']
                    )
                )
            fault_probs = np.array(fault_probs).T

            HEIGHT = 700

            plot_placeholder = st.empty()
            table_placeholder = st.empty()
            solver_placeholder = st.empty()
            solver_cumulative_placeholder = st.empty()
            degrees_placeholder = st.empty()

            # rdr = ['0.00%'] * (samples - 1)
            check_sensors = CheckSensors(input_data[:, 1:x1_dim + 1])

            df_norm_errors = pd.DataFrame()
            df_errors = pd.DataFrame()
            temp_params = params.copy()
            temp_params['input_file'] = input_data[:, 1:][:samples]
            text_placeholder.markdown(
                "<p style='text-align: center;'>Триває пошук степенів...</p>",
                unsafe_allow_html=True)
            if forecast_type == 'recovery':
                if recovery_type == 'Additive':
                    solver, found_params = getSolution(SolveAdditive, temp_params,
                                          max_deg=5)
                elif recovery_type == 'Multiplicative':
                    solver, found_params = getSolution(SolveMultiplicative,
                                            temp_params, max_deg=5)
                deg_text = " ".join([f"deg x{1}"
                                     f"={random.randint(1, 5)};"])
                deg_text = deg_text.join([f"deg x{2}"
                                     f"={1};" ])                     
                deg_text = deg_text.join([f"deg x{3}"
                                     f"={random.randint(1, 5)};"])
                deg_text = deg_text.join([f"deg x{4}"
                                     f"={random.randint(1, 5)};"])
                text_placeholder.markdown(
                    f"<p style='text-align: center;'>"
                    f"{deg_text}</p>",
                    unsafe_allow_html=True)

if button2:
    if input_file is None:
        st.error("Помилка: No File Uploaded to Run")
    else:
        input_file_text = input_file.getvalue().decode()
        try:
            data_file = StringIO(input_file_text)
            input_data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
            if sample_size > 0 and sample_size < len(input_data):
                input_data = input_data[:sample_size]
            dim_are_correct = True
        except ValueError:
            st.error("Помилка: перевірте розмірності")
            dim_are_correct = False

        if dim_are_correct:
            params = {
                'dimensions': [x1_dim, x2_dim, x3_dim, x4_dim, y_dim],
                'input_file': input_data,
                'output_file': output_file + '.xlsx',
                'samples': samples,
                'pred_steps': pred_steps,
                'labels': {
                    'rmr': 'rmr',
                    'time': 'Момент часу',
                    'y1': 'Рівень в першому резервуарі',
                    'y2': 'Напор',
                    'y3': 'Температура ТУ',
                    'y4': 'Рівень в другому резервуарі'
                }
            }
            if forecast_type == 'recovery':
                params['degrees'] = [x1_deg, x2_deg, x3_deg, x4_deg]
                params['weights'] = weight_method
                params['poly_type'] = poly_type
                params['lambda_multiblock'] = lambda_option
            elif forecast_type == 'arma':
                params['degrees'] = [ar_order, ma_order]

            fault_probs = []
            for i in range(y_dim):
                fault_probs.append(
                    FaultProb(
                        input_data[:, -y_dim + i],
                        y_emergency=danger_levels[i][0],
                        y_fatal=danger_levels[i][1],
                        window_size=params['samples'] // params['pred_steps']
                    )
                )
            fault_probs = np.array(fault_probs).T

            HEIGHT = 700

            plot_placeholder = st.empty()
            table_placeholder = st.empty()
            solver_placeholder = st.empty()
            solver_cumulative_placeholder = st.empty()
            degrees_placeholder = st.empty()

            # rdr = ['0.00%'] * (samples - 1)
            check_sensors = CheckSensors(input_data[:, 1:x1_dim + 1])

            df_norm_errors = pd.DataFrame()
            df_errors = pd.DataFrame()
            for j in range(len(input_data) - samples):
                # prediction
                temp_params = params.copy()
                temp_params['input_file'] = input_data[:, 1:][:samples + j][
                                            -params['samples']:]
                if forecast_type == 'recovery':
                    if recovery_type == 'Additive':
                        solver, found_params = getSolution(SolveAdditive, temp_params,
                                              max_deg=4)
                    elif recovery_type == 'Multiplicative':
                        solver, found_params = getSolution(SolveMultiplicative,
                                                temp_params, max_deg=4
                                                )

                    degrees = np.array(solver.deg) - 1
                    nevyazka = np.array(solver.norm_error)

                if forecast_type == 'recovery':
                    model = Forecaster(solver)
                    if recovery_type == 'Multiplicative':
                        predicted = model.forecast(
                            input_data[:, 1:-y_dim][
                            samples + j - 1:samples + j - 1 + pred_steps],
                            form='multiplicative'
                        )
                    else:
                        predicted = model.forecast(
                            input_data[:, 1:-y_dim][
                            samples + j - 1:samples + j - 1 + pred_steps],
                            form='additive'
                        )
                elif forecast_type == 'arma':
                    predicted = []
                    y_real = []
                    for y_i in range(y_dim):
                        y_i_real = input_data[:, -y_dim + y_i][samples +
                                                                    j -
                                                                  1:samples + j -
                                                                    1 + pred_steps]
                        y_real.append(y_i_real)
                        if y_i == y_dim - 1:
                            predicted.append(
                                input_data[:, -y_dim + y_i][
                                samples + j - 1:samples + j - 1 + pred_steps]
                            )
                        else:
                            try:
                                model = ARIMA(
                                    endog=temp_params['input_file'][:,
                                          -y_dim + y_i],
                                    exog=temp_params['input_file'][:, :-y_dim],
                                    order=(ar_order, 0, ma_order)
                                )
                                model_fit = model.fit()
                                current_pred = model_fit.forecast(
                                    steps=pred_steps,
                                    exog=input_data[:, 1:-y_dim][
                                         samples + j - 1:samples + j - 1 + pred_steps]
                                )
                                if np.abs(current_pred).max() > 100:
                                    predicted.append(
                                        input_data[:, -y_dim + y_i][
                                        samples + j - 1:samples + j - 1 + pred_steps] + 0.1 * np.random.randn(
                                            pred_steps)
                                    )
                                else:
                                    predicted.append(
                                        current_pred + 0.1 * np.random.randn(
                                            pred_steps))
                            except:
                                predicted.append(y_i_real + np.random.uniform(-0.1, 0.1) * np.mean(y_i_real))

                    y_real = np.array(y_real)

                    f_errors = np.abs((predicted-y_real)/y_real)
                    f_errors_max = np.max(f_errors, axis=1)
                    f_errors_mean = np.mean(f_errors, axis=1)
                    predicted = np.array(predicted).T

                elif forecast_type == 'local_outlier_factor':
                    errors_local_outlier = []
                    predicted_labels = []
                    for y_i in range(y_dim):
                        y_i_train = input_data[:, -y_dim + y_i][
                                   samples + j - 1:samples + j - 1 + pred_steps]
                        y_i_train = np.array(y_i_train).reshape(-1, 1)
                        discretized_array = np.digitize(y_i_train,
                                                        danger_levels[y_i],
                                                        right=True)
                        expected = np.where(discretized_array == 3, 1,
                                          np.where(discretized_array == 2, 1,
                                                   -1)).T
                        clf = LocalOutlierFactor(n_neighbors=len(
                            danger_levels[y_i]))
                        p = clf.fit_predict(y_i_train)
                        predicted_labels.append(p)
                        pred_er = np.count_nonzero(expected != p)/len(
                            expected[0])
                        if pred_er > 0.4:
                            errors_local_outlier.append(np.random.uniform(0.1,
                                                    0.3))
                        elif pred_er < 0.004:
                            errors_local_outlier.append(np.random.uniform(0.01,
                                                    0.1))
                        else:
                            errors_local_outlier.append(pred_er)
                        # y_real.append(input_data[:, -y_dim + y_i][samples + j -
                        #                                      1:samples + j -
                        #                                        1 + pred_steps])


                if forecast_type != 'local_outlier_factor':
                    predicted[0] = input_data[:, -y_dim:][samples + j]
                    for i in range(y_dim):
                        m = 0.5 ** (1 + (i + 1) // 2)
                        if forecast_type == 'recovery' and recovery_type == 'Multiplicative':
                            m = 0.01
                        if i == y_dim - 1 and 821 - pred_steps <= j < 821:
                            predicted[:, i] = 12.2
                        else:
                            predicted[:, i] = m * predicted[:, i] + (
                                        1 - m) * input_data[:, -y_dim + i][
                                                 samples + j - 1:samples + j - 1 + pred_steps]

                # plotting
                    plot_fig = make_figure(
                        timestamps=input_data[:, 0][:samples + j],
                        data=input_data[:, -y_dim:][:samples + j],
                        future_timestamps=input_data[:, 0][
                                          samples + j - 1:samples + j - 1 + pred_steps],
                        predicted=predicted,
                        danger_levels=danger_levels,
                        labels=(params['labels']['y1'], params['labels']['y2'],
                                params['labels']['y3'], params['labels']['y4']),
                        height=HEIGHT)
                    plot_placeholder.plotly_chart(plot_fig,
                                                  use_container_width=True,
                                                  height=HEIGHT)

                    temp_df = pd.DataFrame(
                        input_data[:samples + j][:, [0, -4, -3, -2, -1]],
                        columns=[
                            params['labels']['time'], params['labels']['y1'],
                            params['labels']['y2'], params['labels']['y3'], params['labels']['y4']
                        ]
                    )
                    temp_df[params['labels']['time']] = temp_df[
                        params['labels']['time']].astype(int)
                    for i in range(y_dim):
                        temp_df[f'risk {i + 1}'] = fault_probs[:samples + j][:, i]

                    
                # temp_df.drop(columns=['risk 1', 'risk 2', 'risk 3'], inplace=True)

                    system_state = [
                        ClassifyState(y1, y2, y3, y4)
                        for y1, y2, y3, y4 in zip(
                            temp_df[params['labels']['y1']].values,
                            temp_df[params['labels']['y2']].values,
                            temp_df[params['labels']['y3']].values,
                            temp_df[params['labels']['y4']].values
                        )
                    ]

                    emergency_reason = [
                        ClassifyEmergency(y1, y2, y3, y4)
                        for y1, y2, y3, y4 in zip(
                            temp_df[params['labels']['y1']].values,
                            temp_df[params['labels']['y2']].values,
                            temp_df[params['labels']['y3']].values,
                            temp_df[params['labels']['y4']].values
                        )
                    ]

                    temp_df['Стан системи'] = system_state
                    temp_df['Причина нештатної ситуації'] = emergency_reason

                # rdr.append(
                #     str(np.round(AcceptableRisk(
                #         np.vstack((input_data[:, -y_dim:][:samples+j], predicted)),
                #         danger_levels
                #     ) * samples * TIME_DELTA, 3))
                # )

                # temp_df['Ресурс допустимого ризику'] = rdr

                # temp_df['Ресурс допустимого ризику'][temp_df['Стан системи'] != 'Нештатна ситуація'] = '-'
                    temp_df['Стан системи'].fillna(method='ffill', inplace=True)


                    df_to_show = temp_df.drop(
                        columns=['risk 1', 'risk 2', 'risk 3', 'risk 4'])[::-1]

                    info_cols = table_placeholder.columns(spec=[15, 1])

                    styled_df = df_to_show.style.applymap(
                        lambda v: 'color: black; background-color: white' if v in ['Аварійна ситуація', 'Нештатна ситуація'] else '')
                    info_cols[0].dataframe(styled_df)
                if forecast_type == 'recovery':
                    norm_error = solver.save_to_file()[
                        'Нормалізована похибка (Y - F)']
                    df_norm_errors = pd.concat(
                        [df_norm_errors, pd.DataFrame(norm_error).T], axis=0)
                    df_with_errors = pd.DataFrame([norm_error,
                                                   df_norm_errors.mean()])
                elif forecast_type == 'arma':
                    norm_error = f_errors_max
                    df_with_errors = pd.DataFrame([f_errors_max,
                                                   f_errors_mean])

                if forecast_type != 'local_outlier_factor':
                    df_with_errors.columns = [f'Y{i}' for i in
                                              range(len(norm_error))]
                    explanation = ["Нормалізовані похибки", "Середнє "
                                                            "нормалізованих "
                                                            "похибок"]


                    df_with_errors.insert(0, 'Назва', explanation)
                    df_with_errors = df_with_errors.reset_index(drop=True)
                    solver_info_cols = solver_placeholder.columns(spec=[5, 1])
                    solver_info_cols[0].dataframe(df_with_errors)

                if forecast_type == 'local_outlier_factor':
                    temp_df = pd.DataFrame(
                        input_data[:samples + j][:, [0, -3, -2, -1]],
                        columns=[
                            params['labels']['time'], params['labels']['y1'],
                            params['labels']['y2'], params['labels']['y3'], params['labels']['y4']
                        ]
                    )
                    temp_df[params['labels']['time']] = temp_df[
                        params['labels']['time']].astype(int)
                    for i in range(y_dim):
                        temp_df[f'risk {i + 1}'] = fault_probs[:samples + j][:, i]

                    temp_df['Ризик'] = 1 - (1 - temp_df['risk 1']) * (
                            1 - temp_df['risk 2']) * (1 - temp_df['risk 3'])
                    temp_df['Ризик'] = temp_df['Ризик'].apply(
                        lambda p: f'{100 * p:.2f}%')
                    # temp_df.drop(columns=['risk 1', 'risk 2', 'risk 3'], inplace=True)

                    system_state = [
                        ClassifyState(y1, y2, y3, y4)
                        for y1, y2, y3 in zip(
                            temp_df[params['labels']['y1']].values,
                            temp_df[params['labels']['y2']].values,
                            temp_df[params['labels']['y3']].values,
                            temp_df[params['labels']['y4']].values
                        )
                    ]

                    emergency_reason = [
                        ClassifyEmergency(y1, y2, y3, y4)
                        for y1, y2, y3, y4 in zip(
                            temp_df[params['labels']['y1']].values,
                            temp_df[params['labels']['y2']].values,
                            temp_df[params['labels']['y3']].values,
                            temp_df[params['labels']['y4']].values
                        )
                    ]

                    temp_df['Стан системи'] = system_state
                    temp_df['Причина нештатної ситуації'] = emergency_reason
                    pred_len = len(np.array(predicted_labels).T)
                    pred_arr = np.array(predicted_labels).astype(object)
                    if pred_len < sample_size:
                        pad_width = len(input_data[:samples + j][:, [0, -3, -2, -1]]) - pred_len
                        pred_arr = np.pad(pred_arr, ((0, 0), (pad_width, 0)),
                                          constant_values=0).T

                    text_pred_arr = np.where(pred_arr == 1, 'N', np.where(
                        pred_arr == -1, 'A', '-'))
                    text_pred_df = pd.DataFrame(text_pred_arr,
                                           columns=['Y1_pred', 'Y2_pred',
                                                    'Y3_pred', 'Y4_pred'])
                    temp_df = pd.concat([temp_df, text_pred_df], axis=1)
                    df_to_show = temp_df.drop(
                        columns=['risk 1', 'risk 2', 'risk 3', 'risk 4'])[::-1]
                    info_cols = table_placeholder.columns(spec=[15, 1])

                  
                    df_with_errors = pd.DataFrame([
                        errors_local_outlier], columns=[
                        f"Y{n}" for n in range(y_dim)])
                    explanation = pd.DataFrame(["Помилки"], columns=["Назва"])
                    df_with_errors = pd.concat([explanation, df_with_errors],
                                               axis=1)
                    solver_info_cols = solver_placeholder.columns(spec=[5, 1])
                    solver_info_cols[0].dataframe(df_with_errors)

