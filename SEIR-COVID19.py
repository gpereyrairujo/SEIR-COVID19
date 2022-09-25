import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


## Códigos del modelo

### Función para correr el modelo

def modelo_SEIR(tabla_datos, fila_inicio=0, fila_fin=-1):

    '''
    Modelo SEIR de paso diario

    Variables de entrada:
        tabla_datos: dataframe con (al menos) las siguientes columnas:
            R0: parámetro R0, un valor para cada fila/día
            Alpha: parámetro Alpha, un valor para cada fila/día
            Gamma: parámetro Gamma, un valor para cada fila/día
            Beta: parámetro Beta (se calcula a partir de R0 y Gamma)
            S: individuos susceptibles, valor para fila inicial
            E: individuos expuestos, valor para fila inicial
            I: individuos infectados, valor para fila inicial
            Ia: individuos infectados acumulados, valor para fila inicial
            R: individuos recuperados/removidos, valor para fila inicial
        fila_inicio: fila correspondiente al día 0 de la simulación (default comienza en fila 0)
        fila_fin: fila correspondiente al último día de la simulación (default hasta la última fila)
    
    Salida: mismo dataframe de entrada con los resultados de individuos S, E, I, R, Ia
    '''

    # hacer una copia de los datos (para no modificar la tabla de datos original)
    datos = tabla_datos.copy()

    # si no se ingresa valor de última fila, contar las filas
    if(fila_fin==-1): fila_fin = len(datos.index) - 1

    for d in range(fila_inicio, fila_fin):

        # leer valores dia actual
        S = datos.loc[d,'S']
        E = datos.loc[d,'E']
        I = datos.loc[d,'I']
        R = datos.loc[d,'R']
        T = S+I+R

        # leer parámetros día actual
        R0 = datos.loc[d,'R0']
        alpha = datos.loc[d,'Alpha']
        gamma = datos.loc[d,'Gamma']
        
        # calcular Beta a partir del R0 y escribirlo en la tabla
        beta = R0 * gamma
        datos.loc[d, 'Beta'] = beta
        
        # calcular valores día siguiente
        S1 = S - beta * S * I/T
        E1 = E + beta * S * I/T - alpha * E
        I1 = I + alpha * E + - gamma * I
        R1 = R + I * gamma
        Ia1 = I1 + R1

        # escribir datos día siguiente
        datos.loc[d+1,'S'] = S1
        datos.loc[d+1,'E'] = E1
        datos.loc[d+1,'I'] = I1
        datos.loc[d+1,'R'] = R1
        datos.loc[d+1,'Ia'] = Ia1

    # resultado
    return datos




### Función para ajustar el modelo, minimizando la diferencia entre valores observados y simulados.

import random
from sklearn.metrics import mean_squared_error
from math import sqrt

def ajustar_modelo(modelo, datos, serie_estimados, serie_observados, inicio_ajuste=0, fin_ajuste=-1, variacion_aleatoria=0.05, maximo_iteraciones=2000, maximo_iteraciones_sin_cambios=300):

    '''
    Función para ajustar parámetros de modelos SEIR de paso diario
    Variables de entrada:
    modelo                          función del modelo a ejecutar
    datos                           tabla de datos con las variables de entrada del modelo
    serie_estimados                 nombre de la serie que contiene los datos estimados por el modelo 
    serie_observados                nombre de la serie que contiene los datos observados
    inicio_ajuste                   fila a partir de la cual se comparan observado y estimado (por defecto la primera fila)
    fin_ajuste                      fila a hasta la cual se comparan observado y estimado (por defecto la última fila)
    variacion_aleatoria             proporción en la que se varían los parámetros en cada iteración
    maximo_iteraciones              cantidad máxima de iteraciones
    maximo_iteraciones_sin_cambios  cantidad de iteraciones sin cambios para considerar ajustado el modelo
    '''

    # hacer una copia de los datos de entrada
    ajuste_preliminar = datos.copy()
    # contar la cantidad de datos observados disponibles
    cantidad_datos = ajuste_preliminar[serie_observados].count()
    ultima_fila = cantidad_datos - 1
    # chequear si se ingresó un valor para la última fila a considerar para el ajuste
    if(fin_ajuste==-1): fin_ajuste=ultima_fila
    # valor inicial rmse
    mejor_rmse = -1
    
    # iterar para mejorar el ajuste, hasta el valor máximo permitido
    for i in range(maximo_iteraciones):

        # correr el modelo para la tabla completa
        ajuste_preliminar = modelo(ajuste_preliminar, 0, ultima_fila)

        # calcular RMSE para el período de ajuste (desde inicio_ajuste hasta fin_ajuste)
        datos_estimados  = ajuste_preliminar.loc[inicio_ajuste:fin_ajuste, serie_estimados]
        datos_observados = ajuste_preliminar.loc[inicio_ajuste:fin_ajuste, serie_observados]
        rmse = sqrt(mean_squared_error(datos_estimados, datos_observados))

        # verificar si el RMSE mejoró (o si es la primera iteración)
        if(rmse<mejor_rmse or i==0): # and minima_diferencia>=0):
            mejor_rmse = rmse
            mejor_ajuste = ajuste_preliminar.copy()
            iteraciones_sin_cambios = 0

        # si no, volver al mejor ajuste anterior
        else:
            ajuste_preliminar = mejor_ajuste.copy()
            iteraciones_sin_cambios = iteraciones_sin_cambios + 1

        # aleatorizar los parámetros para una nueva iteración
        aleatorizar_parametros(ajuste_preliminar, variacion_aleatoria)

        # mostrar progreso
        if(i % 100)==0:
            print(".", end=" ")

        # si ya hubo muchas iteraciones sin que haya cambios, terminar el ajuste
        if(iteraciones_sin_cambios > maximo_iteraciones_sin_cambios): break

    # devolver como resultado el mejor mejor_ajuste obtenido
    return mejor_ajuste


def aleatorizar_parametros(datos, variacion_aleatoria):

    # leer los valores actuales de los parámetros
    R0  = datos.loc[0,'R0']
    S   = datos.loc[0,'S']
    E   = datos.loc[0,'E']
    I   = datos.loc[0,'I']
    Ia  = datos.loc[0,'Ia']
    R   = datos.loc[0,'R']
    T   = S + E + I + R
    
    # aleatorizar el parámetro R0
    multiplicador = random.uniform(1-variacion_aleatoria, 1+variacion_aleatoria)
    R0  = R0  * multiplicador

    # aleatorizar los valores de E e I
    multiplicador = random.uniform(1-variacion_aleatoria, 1+variacion_aleatoria)
    E  = E * multiplicador
    I  = I * multiplicador

    # aleatorizar el valor de R
    #multiplicador = random.uniform(1-variacion_aleatoria, 1+variacion_aleatoria)
    #R = R * multiplicador
    
    # recalcular el resto de las variables iniciales
    #Ia = R + I
    R = Ia - I
    S = T - E - I - R

    # actualizar los valores en la planilla de datos
    datos['R0']        = R0
    datos.loc[0, 'S']  = S
    datos.loc[0, 'E']  = E
    datos.loc[0, 'I']  = I
    datos.loc[0, 'Ia'] = Ia
    datos.loc[0, 'R']  = R


def proyeccion(datos, columna_fecha, columna_datos, fecha_proyeccion, carpeta, periodo_latencia=3, periodo_infectivo=7, media_movil=False, variacion_diaria=0.015):

    # función para estimar e incorporar los valores iniciales a la tabla antes del ajuste 
    def valores_iniciales_ajuste(datos_ajuste, periodo_latencia=3, periodo_infectivo=7):

        # parámetros del modelo SEIR
        R0 = 1.5
        alpha = 1 / periodo_latencia
        gamma = 1 / periodo_infectivo
        beta = R0 * gamma
        # población total
        T = 650000

        # agregar columnas para los parámetros del modelo, e incluir los valores iniciales en todas las filas
        ## valores de R0 y Beta a partir de los resultados del ajuste con nexo/sin nexo
        datos_ajuste['R0']    = R0
        datos_ajuste['Alpha'] = alpha
        datos_ajuste['Beta']  = beta
        datos_ajuste['Gamma'] = gamma

        # estimar valores iniciales de casos expuestos y activos
        casos_primera_semana = datos_ajuste.loc[7, columna_datos] - datos_ajuste.loc[0, columna_datos]
        I = casos_primera_semana
        E = I * gamma / alpha

        # tomar valores iniciales de casos acumulados
        Ia  = datos_ajuste.loc[0, columna_datos]
        R = Ia - I
        S = T - E - I - R

        # agregar columnas para los compartimientos del modelo y los valores iniciales en la primera fila
        datos_ajuste.loc[0, 'S']     = S
        datos_ajuste.loc[0, 'E']     = E
        datos_ajuste.loc[0, 'I']     = I
        datos_ajuste.loc[0, 'Ia']    = Ia
        datos_ajuste.loc[0, 'R']     = R


    # estilo para los gráficos
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = [0.93,0.93,0.93]
    
    # parámetros de duración de los ajustes y la proyección
    duraciones_ajustes = [21, 14, 7]
    duracion_lag = 1    # cuántos días antes se inicia la simulación para que se 'estabilicen' los valores
    duracion_proyeccion = 21
    fecha_fin_proyeccion = fecha_proyeccion + pd.Timedelta(str(duracion_proyeccion) + ' days')
    duracion_proyeccion_media_movil = duracion_proyeccion + 3       # aumentar la duración de la proyección para compensar el atraso generado por la media móvil
    fecha_fin_proyeccion_media_movil = fecha_proyeccion + pd.Timedelta(str(duracion_proyeccion_media_movil) + ' days')

    # suavización de los datos (media móvil de 7 días)
    columna_datos_originales = columna_datos
    datos['fecha_media_movil'] = datos[columna_fecha] - pd.Timedelta('3 days')
    datos['media_movil'] = datos[columna_datos_originales].rolling(7).mean()
    if media_movil:
        columna_datos = 'media_movil'

    # seleccionar los datos hasta la fecha de fin de proyección
    if media_movil:
        datos_recortados = datos.loc[datos[columna_fecha]<=fecha_fin_proyeccion_media_movil].reset_index(drop=True)
    else:
        datos_recortados = datos.loc[datos[columna_fecha]<=fecha_fin_proyeccion].reset_index(drop=True)

    # texto para incluir en los archivos
    if media_movil: 
        mm_str = '-media-movil'
    else:
        mm_str = ''

    # hacer los ajustes por períodos
    for periodo, duracion_ajuste in enumerate(duraciones_ajustes):

        # recortar la tabla de datos y dejar sólo las filas que se van a usar para el ajuste
        if media_movil:
            filas = duracion_lag + duracion_ajuste + duracion_proyeccion_media_movil + 1
        else:
            filas = duracion_lag + duracion_ajuste + duracion_proyeccion + 1
        datos_ajuste = datos_recortados.tail(filas).reset_index(drop=True)

        # agregar datos iniciales
        valores_iniciales_ajuste(datos_ajuste, periodo_latencia, periodo_infectivo)

        # correr la función de ajuste de parámetros
        resultado_ajuste = ajustar_modelo(modelo_SEIR, datos_ajuste, 'Ia', columna_datos, inicio_ajuste=duracion_lag, fin_ajuste=duracion_lag+duracion_ajuste, variacion_aleatoria=0.05)
        resultado_proyeccion = resultado_ajuste.tail(duracion_proyeccion+1).reset_index(drop=True)
        valor_final_proyeccion = resultado_proyeccion.loc[len(resultado_proyeccion.index)-1,'Ia']

        # mostrar R0 ajustado
        print('R0 ajustado ' + str(duracion_ajuste) + 'd' + mm_str.replace('-',' ') + ':', round(resultado_ajuste.loc[0, 'R0'], 2))

        # guardar resultados del ajuste
        archivo = fecha_proyeccion.strftime('%Y-%m-%d') + '_ajuste-' + str(duracion_ajuste) + 'd' + '_latencia-' + str(periodo_latencia) + '_infectivo-' + str(periodo_infectivo) + mm_str + '.csv'
        resultado_ajuste.to_csv(carpeta + archivo)

        # identificar máximo y mínimo
        if periodo == 0: 
            maximo = minimo = valor_final_proyeccion
            maximo_proyeccion = minimo_proyeccion = resultado_proyeccion.copy()
        else:
            if valor_final_proyeccion > maximo: 
                maximo = valor_final_proyeccion
                maximo_proyeccion = resultado_proyeccion.copy()
            if valor_final_proyeccion < minimo: 
                minimo = valor_final_proyeccion
                minimo_proyeccion = resultado_proyeccion.copy()

    ### VARIAR R0
             
    # aumentar R0
    R0 = maximo_proyeccion.loc[0,'R0']
    maximo_proyeccion_aumento_R0          = maximo_proyeccion.copy()
    maximo_proyeccion_aumento_R0['R0']    = ( 1 + pd.Series(range(len(maximo_proyeccion.index))) * variacion_diaria ) * R0
    maximo_proyeccion_aumento_R0          = modelo_SEIR(maximo_proyeccion_aumento_R0)
    # disminuir R0
    R0 = minimo_proyeccion.loc[0,'R0']
    minimo_proyeccion_disminucion_R0          = minimo_proyeccion.copy()
    minimo_proyeccion_disminucion_R0['R0']    = ( 1 - pd.Series(range(len(maximo_proyeccion.index))) * variacion_diaria ) * R0
    minimo_proyeccion_disminucion_R0          = modelo_SEIR(minimo_proyeccion_disminucion_R0)

    # ver los valores de casos estimados finales en cada simulación
    ultima_fila = len(minimo_proyeccion_disminucion_R0.index) -1
    valor_final_minimo = minimo_proyeccion_disminucion_R0.loc[ultima_fila,'Ia']
    valor_final_maximo = maximo_proyeccion_aumento_R0.loc[ultima_fila,'Ia']

    # guardar resultados de las proyecciones
    archivo = fecha_proyeccion.strftime('%Y-%m-%d') + '_variacion-diaria-' + str(round(variacion_diaria*100,1)) + '_latencia-' + str(periodo_latencia) + '_infectivo-' + str(periodo_infectivo) + '_proyeccion-minima' + mm_str + '.csv'
    minimo_proyeccion_disminucion_R0.to_csv(carpeta + archivo)
    archivo = fecha_proyeccion.strftime('%Y-%m-%d') + '_variacion-diaria-' + str(round(variacion_diaria*100,1)) + '_latencia-' + str(periodo_latencia) + '_infectivo-' + str(periodo_infectivo) + '_proyeccion-maxima' + mm_str + '.csv'
    maximo_proyeccion_aumento_R0.to_csv(carpeta + archivo)

    ### GRAFICAR RESULTADO

    plt.figure(figsize=(4.2, 3))

    # datos observados (los días previos que se usan para ajustar y los días posteriores para los que se proyecta)
    x_min = fecha_proyeccion - pd.Timedelta( str(max(duraciones_ajustes)-1) + ' days' )
    x_max = fecha_proyeccion + pd.Timedelta( str(duracion_proyeccion) + ' days' ) 
    datos_para_grafico = datos.loc[(datos[columna_fecha]<=x_max) & (datos[columna_fecha]>=x_min)].reset_index(drop=True)

    # línea vertical inicio proyección
    plt.axvline(fecha_proyeccion, color=[0.75, 0.75, 0.75], zorder=2)

    # casos estimados variando R0
    if media_movil: 
        columna_fecha_grafico = 'fecha_media_movil'
    else:
        columna_fecha_grafico = columna_fecha
    plt.fill_between(
        maximo_proyeccion[columna_fecha_grafico], 
        minimo_proyeccion_disminucion_R0['Ia'], 
        maximo_proyeccion_aumento_R0['Ia'], 
        color=[0.39,0.62,0.83], #[0.99,0.81,0.8],
        label='Casos proyectados',
        zorder=1)

    # datos originales
    plt.plot(
        datos_para_grafico[columna_fecha], 
        datos_para_grafico[columna_datos_originales],
        color=[0.1,0.1,0.1], #[0.13,0.36,0.75], #[0.96,0.07,0.01], 
        solid_capstyle='projecting',
        label="Casos reportados",
        zorder=3)

    # ubicación leyenda
    plt.legend(loc='upper left', prop={'size':9})
    # formato de fecha
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(6))
    # ejes
    plt.xlim(x_min, x_max)
    plt.gca().set_xlabel('Fecha')
    plt.gca().set_ylabel('Casos')

    # guardar figura
    archivo = fecha_proyeccion.strftime('%Y-%m-%d') + '_variacion-diaria-' + str(round(variacion_diaria*100,1)) + '_latencia-' + str(periodo_latencia) + '_infectivo-' + str(periodo_infectivo) + mm_str + '.png'
    plt.savefig(carpeta+archivo, bbox_inches = 'tight', pad_inches = 0.1, dpi=150)
    plt.show()
    plt.close('all')
    

# salida de ejemplo

# datos observados
archivo_datos_observados = './datos/casos_mgp.csv'
columna_fecha = 'fecha'
columna_datos = 'confirmados_minsal'
datos = pd.read_csv(archivo_datos_observados, parse_dates=[0])

# ajustar modelo
fecha_proyeccion = pd.to_datetime('2020-11-19')
print('Proyección para la fecha:', fecha_proyeccion)
carpeta_resultados = './salida_ejemplo/'
proyeccion(datos, columna_fecha, columna_datos, fecha_proyeccion, carpeta_resultados, media_movil=False)

