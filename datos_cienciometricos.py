# Libreria de python para extraer datos de cvlac y gruplac mediante webscraping.
# Se extrae informacion de grupos, integrantes de los grupos, informacion de perfiles de los grupos
# y informacion de la produccion cientifica 
# ---------------------------------------------------------------------------------------------------
# ELaborado por: Luis A. Valencia - correo: lavalenciah12@gmail.com - 08/agosto/2022

# LIBRERIAS 
from inspect import stack
import aiohttp
import asyncio
import nest_asyncio
nest_asyncio.apply()
import pandas as pd 
import re 
import numpy as np 
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stop_words = stopwords.words('spanish') +  stopwords.words('english')
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import requests
from async_class import AsyncClass

# ----------------------------------------------------------------------------------------------
# --------- FUNCIONES AUXILIARES ---------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def convert2df(list_json): 
    """Funcion para convertir Lista de objetos json a dataframe 
    Args:
        list_json (lista de objetos json): lista de objetos json

    Returns:
        pandas dataframe : retorna un dataframe de pandas 
    """
    if len(list_json) == 0:
        return None
    keys    = list_json[0].keys()
    df_dict = {key:[] for key in keys}
    for val in list_json: 
        for key in keys: 
            try:
                df_dict[key].append(val[key])
            except: 
                df_dict[key].append('')
    return pd.DataFrame(df_dict)

def convertdf2json(df): 
    """Convierte un dataframe a una lista de objetos json 

    Args:
        df (pandas dataframe): pandas dataframe 

    Returns:
        list: lista de objetos json 
    """
    list_json = [row.to_dict() for idx, row in df.iterrows()]
    return list_json


# ----------------------------------------------------------------------------------------------
# ---------     PREPROCESAMIENTO DE LOS DATOS --------------------------------------------------
# ----------------------------------------------------------------------------------------------

async def formato_perfiles_grupos(info_perfiles): 
    """_summary_

    Args:
        info_perfiles (list): lista de objetos json 

    Returns:
        list : lista de objetos json 
    """

    investigadores_df         = [] 
    colaboracion_df           = [] 
    trayectoria_df            = [] 
    estabilidad_produccion_df = [] 
    generacion_nuevo_df       = [] 
    desarrollo_tec_df         = [] 
    apropiacion_social_df     = [] 
    recurso_humano_df         = []

    for dato in info_perfiles: 
        nombre_grupo = dato['grupo']
        # Perfil de investigadores 
        try: 
            df3 = pd.DataFrame(dato['Perfil_integrantes'])
            investigadores = [{'nombre_grupo':nombre_grupo, 'Indicador': df3.iloc[i, 0],'Minimo':df3.iloc[i, 3], 'Cuartil 4' : df3.iloc[i, 4], 
                        'Cuartil 3': df3.iloc[i, 5],'Cuartil 2' : df3.iloc[i, 6], 
                        'Maximo':df3.iloc[i, 7], 'Valor indicador':df3.iloc[i, 9], 
                        'Cuartil ubicacion': df3.iloc[i, 10],} for i in range(2,14)]
        except: 
            investigadores = {'nombre_grupo':nombre_grupo, 'Indicador': np.NAN,'Minimo':np.NAN, 'Cuartil 4' : np.NAN, 
                        'Cuartil 3': np.NAN,'Cuartil 2' :np.NAN, 
                        'Maximo':np.NAN, 'Valor indicador':np.NAN, 
                        'Cuartil ubicacion': np.NAN}
            #print('investigadores')
            #break
        investigadores_df.append(investigadores)
        # Perfil de colaboracion 
        try: 
            df3   = pd.DataFrame(dato['Colaboracion'])
            colaboracion    = [{'nombre_grupo':nombre_grupo, 'Indicador': df3.iloc[i, 0],'Minimo':df3.iloc[i, 3], 'Cuartil 4' : df3.iloc[i, 4], 
                        'Cuartil 3': df3.iloc[i, 5],'Cuartil 2' : df3.iloc[i, 6], 
                        'Maximo':df3.iloc[i, 7], 'Valor indicador':df3.iloc[i, 9], 
                        'Cuartil ubicacion': df3.iloc[i, 10],} for i in range(2,4)]
        except: 
            colaboracion    = {'nombre_grupo':nombre_grupo, 'Indicador': np.NAN,'Minimo':np.NAN, 'Cuartil 4' : np.NAN, 
                        'Cuartil 3': np.NAN,'Cuartil 2' :np.NAN, 
                        'Maximo':np.NAN, 'Valor indicador':np.NAN, 
                        'Cuartil ubicacion': np.NAN}
        colaboracion_df.append(colaboracion)

        # Trayectoria, permamencia y estabilidad 
        try: 
            df3 = pd.DataFrame(dato['Trayectoria_permanencia_estabilidad'])
            trayectoria   = [{'nombre_grupo':nombre_grupo, 'Indicador': df3.iloc[i, 0],'Minimo':df3.iloc[i, 3], 'Cuartil 4' : df3.iloc[i, 4], 
                        'Cuartil 3': df3.iloc[i, 5],'Cuartil 2' : df3.iloc[i, 6], 
                        'Maximo':df3.iloc[i, 7], 'Valor indicador':df3.iloc[i, 9], 
                        'Cuartil ubicacion': df3.iloc[i, 10],} for i in range(2,4)]
        except: 
            trayectoria = {'nombre_grupo':nombre_grupo, 'Indicador': np.NAN,'Minimo':np.NAN, 'Cuartil 4' : np.NAN, 
                        'Cuartil 3': np.NAN,'Cuartil 2' :np.NAN, 
                        'Maximo':np.NAN, 'Valor indicador':np.NAN, 
                        'Cuartil ubicacion': np.NAN}
        trayectoria_df.append(trayectoria)
        # INdicador de estabilidad de la produccion
        try: 
            df5 = pd.DataFrame(dato['Estabilidad_produccion'])
            eprod = {'nombre_grupo':nombre_grupo, 'Indicador': df5.iloc[2, 0] ,  'Minimo':df5.iloc[2, 3], 'Cuartil 4' : df5.iloc[2, 4], 'Cuartil 3': df5.iloc[2, 5],
                    'Cuartil 2' : df5.iloc[2, 6], 'Maximo':df5.iloc[2, 7], 'Valor indicador':df5.iloc[2, 9],  'Cuartil ubicacion': df5.iloc[2, 10]}
        except: 
            eprod = {'nombre_grupo':nombre_grupo, 'Indicador': np.NAN,'Minimo':np.NAN, 'Cuartil 4' : np.NAN, 
                        'Cuartil 3': np.NAN,'Cuartil 2' :np.NAN, 
                        'Maximo':np.NAN, 'Valor indicador':np.NAN, 
                        'Cuartil ubicacion': np.NAN}
        estabilidad_produccion_df.append(eprod)
        # Indicador de generacion del nuevo conocimiento 
        try:
            df6 = pd.DataFrame(dato['Generacion_nuevo_conocimiento'])
            generacion_nuevo = [{'nombre_grupo':nombre_grupo, 'Subtipo_producto': df6.iloc[i, 0],'Minimo':df6.iloc[i, 3], 
                        'Cuartil 4' : df6.iloc[i, 4], 'Cuartil 3': df6.iloc[i, 5],
                        'Cuartil 2' : df6.iloc[i, 6], 'Maximo':df6.iloc[i, 7], 'Valor indicador':df6.iloc[i, 9], 
                        'Cuartil ubicacion':df6.iloc[i, 10]} for i in range(2,13)]
        except: 
            generacion_nuevo = {'nombre_grupo':nombre_grupo, 'Subtipo_producto': np.NAN,'Minimo':np.NAN, 'Cuartil 4' : np.NAN, 
                        'Cuartil 3': np.NAN,'Cuartil 2' :np.NAN, 
                        'Maximo':np.NAN, 'Valor indicador':np.NAN, 
                        'Cuartil ubicacion': np.NAN}
        generacion_nuevo_df.append(generacion_nuevo)
        # Indicador de desarrollo tecnologico 
        try: 
            df3   = pd.DataFrame(dato['Desarrollo_tecnologico'])
            desarrollo_tec    = [{'nombre_grupo':nombre_grupo, 'Indicador': df3.iloc[i, 0],'Minimo':df3.iloc[i, 3], 'Cuartil 4' : df3.iloc[i, 4], 
                        'Cuartil 3': df3.iloc[i, 5],'Cuartil 2' : df3.iloc[i, 6], 
                        'Maximo':df3.iloc[i, 7], 'Valor indicador':df3.iloc[i, 9], 
                        'Cuartil ubicacion': df3.iloc[i, 10],} for i in range(2,7)]
        except: 
            desarrollo_tec = {'nombre_grupo':nombre_grupo, 'Indicador': np.NAN,'Minimo':np.NAN, 'Cuartil 4' : np.NAN, 
                        'Cuartil 3': np.NAN,'Cuartil 2' :np.NAN, 
                        'Maximo':np.NAN, 'Valor indicador':np.NAN, 
                        'Cuartil ubicacion': np.NAN}
        desarrollo_tec_df.append(desarrollo_tec)
        # Indicador de apropiacion social del conocimiento 
        try: 
            df3   = pd.DataFrame(dato['Apropiacion_social_conocimiento'])
            apropiacion_social = [{'nombre_grupo':nombre_grupo, 'Indicador': df3.iloc[i, 0],'Minimo':df3.iloc[i, 3], 'Cuartil 4' : df3.iloc[i, 4], 
                        'Cuartil 3': df3.iloc[i, 5],'Cuartil 2' : df3.iloc[i, 6], 
                        'Maximo':df3.iloc[i, 7], 'Valor indicador':df3.iloc[i, 9], 
                        'Cuartil ubicacion': df3.iloc[i, 10],} for i in range(2,6)]
        except: 
            apropiacion_social = {'nombre_grupo':nombre_grupo, 'Indicador': np.NAN,'Minimo':np.NAN, 'Cuartil 4' : np.NAN, 
                        'Cuartil 3': np.NAN,'Cuartil 2' :np.NAN, 
                        'Maximo':np.NAN, 'Valor indicador':np.NAN, 
                        'Cuartil ubicacion': np.NAN}
        apropiacion_social_df.append(apropiacion_social)
        # Indicador de recurso humano
        try: 
            df3   = pd.DataFrame(dato['formacion_recurso_humano'])
            recurso_humano = [{'nombre_grupo':nombre_grupo, 'Indicador': df3.iloc[i, 0],'Minimo':df3.iloc[i, 3], 'Cuartil 4' : df3.iloc[i, 4], 
                            'Cuartil 3': df3.iloc[i, 5],'Cuartil 2' : df3.iloc[i, 6], 
                            'Maximo':df3.iloc[i, 7], 'Valor indicador':df3.iloc[i, 9], 
                            'Cuartil ubicacion': df3.iloc[i, 10],} for i in range(2,11)]
        except: 
            recurso_humano = {'nombre_grupo':nombre_grupo, 'Indicador': np.NAN,'Minimo':np.NAN, 'Cuartil 4' : np.NAN, 
                        'Cuartil 3': np.NAN,'Cuartil 2' :np.NAN, 
                        'Maximo':np.NAN, 'Valor indicador':np.NAN, 
                        'Cuartil ubicacion': np.NAN}
        recurso_humano_df.append(recurso_humano) 

    return({'investigadores':investigadores_df, 'colaboracion': colaboracion_df, 'trayectoria': trayectoria_df, 'estabilidad_produccion':estabilidad_produccion_df,
           'generacion_nuevo_conocimiento': generacion_nuevo_df, 'desarrollo_tec': desarrollo_tec_df, 'apropiacion_social' : apropiacion_social_df, 'recurso_humano':recurso_humano_df})

async def formato_articulos(articulos, grupo): 
    """Funcion para organizar los objetos de tipo articulo

    Args:
        articulos (list): lista de articulos json
        grupo (str): nombre del grupo al que pertenecen los grupos 

    Returns:
        list: lista de objetos json formateados 
    """

    # Extraccion de titulo
    articulos_fortmateados = []
    for articulo in articulos: 
        try:
            inicio_titulo = re.search(':',articulo).end()
            fin_titulo    = re.search('  ',articulo).start()
            titulo = articulo[inicio_titulo:fin_titulo].replace('"','').replace('.','').title().strip()
        except:
            titulo = 'NA'
        # Extraccion del DOI
        try:
            doi = re.findall('(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)',articulo)[0]
        except:
            doi = 'NA'
        # Extraccion del ISSN 
        try: 
            issn = re.sub('-','',re.findall('\d{4}-\d{3}[\d|X]',articulo)[0])
        except:
            issn = 'NA'
        # Extraccion de autores 
        try: 
            inicio_autores = re.search('Autores:',articulo).end()
            autores = articulo[inicio_autores:].strip().title()
        except:
            autores = 'NA'
            
        # Extraccion del año 
        try:
            anio = re.sub(',','',re.findall(' \d{4} ',articulo)[0],).strip()
        except: 
            anio = 'NA'
            
        # Extraccion del tipo de articulo 
        try:
            inicio = re.search('-',articulo).end()
            fin    = re.search(':',articulo).start()
            tipo_articulo = articulo[inicio:fin].strip().capitalize()
        except:
            tipo_articulo = 'NA'
        info = {'titulo':titulo, 'doi':doi, 'issn':issn, 'autores':autores, 'tipo_articulo':tipo_articulo, 'anio':anio, 'grupo':grupo}

        articulos_fortmateados.append(info)
    if len(articulos_fortmateados) == 0: 
        return articulos_fortmateados
    # Limpieza de articulos 
    articulos_df      = convert2df(articulos_fortmateados).drop_duplicates('titulo')
    docs              = list(articulos_df['titulo'].values)
    vect              = TfidfVectorizer(stop_words= stop_words)
    X1                = vect.fit_transform(docs)
    distance          = (X1 * X1.T).toarray()
    distance          = distance - np.eye(distance.shape[1])
    mask              = (distance >= 0.75) #& (distance < 0.90)
    idx               = np.where(mask)
    distancias_up     = distance[idx]
    posibles          = set()
    count_borrados    = 0
    articulos_limpios = set()
    borrar_articulos  = set()

    for dist in distancias_up:   
        articulos_repetidos = np.array([docs[val] for val in idx[0]])[np.where(np.isclose(dist, distance[idx]))]
        art1 = articulos_df[articulos_df.titulo == articulos_repetidos[0]].index[0]
        art2 = articulos_df[articulos_df.titulo == articulos_repetidos[1]].index[0]
        posibles.add((art1, art2))
        years = articulos_df.loc[[art1,art2], :]['anio'].values
        if years[0] == years[1]: 
            borrar = articulos_df.loc[[art1,art2], :].sort_values(by = 'doi', na_position = 'last').index
            if borrar[0] not in articulos_limpios:
                count_borrados += 1
                articulos_limpios.add(borrar[0])
                borrar_articulos.add(borrar[1])

    articulos_formateados_limpios = convertdf2json(articulos_df.loc[set(articulos_df.index).difference(borrar_articulos), :])

    return(articulos_formateados_limpios)


# Funcion para extraer informacion de libros  
async def formato_libros(libros : str, grupo : str):
    """Funcion para organizar los objetos tipo libro 

    Args:
        libros (list): lista de libros en formato json
        grupo (str): nombre del grupo al que pertenecen los grupos 

    Returns:
        list: lista de objetos json formateados 
    """

    libros_formateados = []
    for libro  in libros :
        # Extraccion del titulo 
        try:
            inicio_titulo = re.search(':',libro).end()
            fin_titulo = re.search('  ',libro).start()
            Titulo = libro[inicio_titulo:fin_titulo].strip().capitalize()
        except: 
            Titulo = 'NA'
            
        # Extraccion de Autores
        try:
            inicio_autores = re.search('Autores:',libro).end()
            Autores = libro[inicio_autores:].strip().title()
        except:
            Autores = 'NA'
        
        # Extraccion de ISBN
        try:
            inicio = re.search('ISBN:', libro).end()
            fin    = re.search('-\d{1,2},', libro).end() - 1
            ISBN = libro[inicio:fin].strip()
        except:
            ISBN = 'NA'
        
        # Extraccion del año 
        try:
            Year = re.sub(',','',re.findall('\d{4}',libro)[0],).strip()
        except:
            Year = 'NA'
        
        # Extraccion del tipo de libro
        try:
            inicio     = re.search('-',libro).end()
            fin        = re.search(':',libro).start()
            Tipo_libro = libro[inicio:fin].strip().capitalize()
        except:
            Tipo_libro = 'NA'
        info = {'titulo':Titulo,'autores':Autores, 'tipo_libro':Tipo_libro, 'anio':Year, 'isbn': ISBN, 'grupo':grupo}   
        libros_formateados.append(info)

    if len(libros_formateados) == 0: 
        return libros_formateados
    
    libros_formateados_json = convertdf2json(convert2df(libros_formateados).drop_duplicates('titulo'))
    return(libros_formateados_json)


async def formato_innovaciones(innovaciones, grupo):
    """Funcion para organizar los objetos tipo innovaciones  

    Args:
        innovaciones (list): lista de innovaciones en formato json 
        grupo (str): nombre del grupo al que pertenecen los grupos 

    Returns:
        list: lista de objetos json formateados 
    """
    
    innovaciones_formateadas = []

    for innovacion in innovaciones: 
        # Extraccion de tipo 
        try: 
            inicio = re.search('.-', innovacion).end()
            fin    = re.search(' : ', innovacion).start()
            tipo = innovacion[inicio:fin].strip().capitalize()
        except: 
            tipo.append('NA')
        # Extraccion de titulo
        try: 
            inicio = re.search(' : ', innovacion).end()
            fin    = re.search('[1,2]\d{3}', innovacion).start()
            titulo = innovacion[inicio:fin].replace(',','').strip().capitalize()
        except: 
            titulo.append('NA')
        # Extraccion ano
        try: 
            ano = re.findall(r'[1,2]\d{3}',innovacion)[0]
        except: 
            ano = 'NA'
        # Extraccion de disponibilidad 
        try: 
            inicio = re.search('Disponibilidad:', innovacion).end()
            fin    = re.search('Institución financiadora:', innovacion).start()
            Disponibilidad = innovacion[inicio:fin].replace(',','').strip().capitalize()
        except: 
            Disponibilidad = 'NA'
        # Extraccion de institucion  
        try: 
            inicio = re.search('Institución financiadora:', innovacion).end()
            fin    = re.search('Autores:', innovacion).start()
            Institucion = innovacion[inicio:fin].replace(',','').strip().capitalize()
        except: 
            Institucion = 'NA'
        # Extraccion de autores   
        try: 
            inicio = re.search('Autores:', innovacion).end()
            Autores = innovacion[inicio:].replace(',','').strip().title()
        except: 
            Autores = 'NA'

        info = {'tipo':tipo, 'titulo':titulo, 'anio':ano, 'autores':Autores,'disponibilidad':Disponibilidad, 'grupo': grupo, 'institucion':Institucion, 'autores':Autores}    
        innovaciones_formateadas.append(info)  
    if len(innovaciones_formateadas) == 0: 
        return  innovaciones_formateadas
    innovaciones_formateadas = convertdf2json(convert2df(innovaciones_formateadas).drop_duplicates('titulo'))
    return(innovaciones_formateadas)



async def formato_trabajos_grado(trabajos_grado, grupo): 
    """Funcion para organizar los objetos tipo trabajos de grado   

    Args:
        trabajos de grado (list): lista de trabajos de grado en formato json 
        grupo (str): nombre del grupo al que pertenecen los grupos 

    Returns:
        list: lista de objetos json formateados 
    """
    trabajos_grado_formateados = []
    for trabajo in trabajos_grado: 
        try: 
            inicio_titulo = re.search(' : ',trabajo).end()
            fin_titulo = re.search('  ', trabajo).start()
            Titulo = trabajo[inicio_titulo:fin_titulo].strip().capitalize()
        except: 
            Titulo = 'NA'

        try: 
            inicio_autor = re.search('Nombre del estudiante:',trabajo).end()
            fin_autor = re.search(',  Programa', trabajo).start()
            Autores = trabajo[inicio_autor:fin_autor].strip().title()
        except: 
            Autores = 'NA'

        try: 
            year = re.findall('\d{4}', trabajo)
            if len(year) > 1: 
                Desde = year[0]
                Hasta = year[1]
            elif len(year) == 1: 
                Desde = year[0]
                Hasta = 'NA'
        except: 
            Desde = 'NA'
            Hasta = 'NA'
            
        try: 
            inicio = re.search('.-',trabajo).end()
            fin    = re.search(' : ',trabajo).start()
            Tipo_producto = trabajo[inicio:fin].strip().capitalize()
        except: 
            Tipo_producto = 'NA'
            
        try: 
            inicio = re.search('Cotutor',trabajo).end()
            tutor = trabajo[inicio+5:].strip().capitalize()
        except:
            tutor = 'NA'
            
        try: 
            inicio = re.search('Institución:',trabajo).end()
            fin    = re.search('Cotutor',trabajo).start()
            institucion = trabajo[inicio:fin-10].strip().capitalize()
        except: 
            institucion = 'NA'
            
        try: 
            inicio = re.search('Programa académico:',trabajo).end()
            fin    = re.search('Número de páginas:',trabajo).start()
            programa_academico = trabajo[inicio:fin].strip().capitalize()
        except: 
            programa_academico = 'NA'
            
        try: 
            inicio = re.search('Número de páginas:',trabajo).end()
            fin    = re.search(', Valoración:',trabajo).start()
            Numero_paginas = trabajo[inicio:fin].strip() 
        except: 
            Numero_paginas = 'NA'
        info = {'titulo':Titulo,'autores':Autores, 'desde':Desde, 'hasta':Hasta, 'tipo_producto': Tipo_producto, 'grupo':grupo,
                'tutor':tutor, 'tipo_producto':Tipo_producto, 'institucion':institucion, 'programa_academico':programa_academico, 
                'numero_paginas':Numero_paginas}     
        trabajos_grado_formateados.append(info)      
    if len(trabajos_grado_formateados) == 0: 
        return trabajos_grado_formateados

    trabajos_grado_formateados = convertdf2json(convert2df(trabajos_grado_formateados).drop_duplicates('titulo'))   
    return(trabajos_grado_formateados)

async def formato_softwares(softwares, grupo): 
    """Funcion para organizar los objetos tipo softwares

    Args:
        softwares (list) : lista de softwares en formato json 
        grupo (str) : nombre del grupo al que pertenecen los grupos 

    Returns:
        list: lista de objetos json formateados 
    """
    softwares_formateados = []
    for software in softwares: 
        # TIpo software 
        try: 
            inicio = re.search('\d{1}.-',software).end() 
            fin    = re.search(' : ', software).start()
            Tipo   = software[inicio:fin].strip().capitalize()
        except: 
            Tipo = 'NA'

        # Titulo 
        try: 
            inicio_titulo = re.search(' : ',software).end()
            fin_titulo = re.search('  ', software).start()
            Titulo = software[inicio_titulo:fin_titulo].strip().capitalize()
        except: 
            Titulo = 'NA'
        # Autores 
        try: 
            inicio_autor = re.search('Autores:',software).end()
            Autores = software[inicio_autor:].strip().capitalize()
        except: 
            Autores = 'NA'

        # Ano 
        try: 
            year = re.findall('\d{4}', software)[0]
        except: 
            year = 'NA'
            
        # Disponibilidad
        try: 
            inicio = re.search('Disponibilidad:',software).end()
            fin    = re.search('Sitio web:',software).start()
            Disponibilidad = software[inicio:fin].strip().replace(',','').capitalize()
        except: 
            Disponibilidad = 'NA'
        
        # Sitio web 
        try: 
            inicio = re.search('Sitio web:',software).end()
            fin    = re.search('Nombre comercial:',software).start()
            Sitio_web = software[inicio:fin].strip().capitalize()
        except:
            Sitio_web = 'NA'
            
        try: 
            inicio = re.search('Nombre comercial:', software).end()
            fin    = re.search('Nombre del proyecto:',software).start()
            Nombre_comercial = software[inicio:fin-10].strip().capitalize()
        except: 
            Nombre_comercial = 'NA'
            
        try: 
            inicio = re.search('Nombre del proyecto:',software).end()
            fin    = re.search('Institución financiadora:',software).start()
            Nombre_proyecto = software[inicio:fin].strip().capitalize()
        except: 
            Nombre_proyecto = 'NA'
            
        try: 
            inicio = re.search('Institución financiadora:',software).end()
            fin    = re.search('Autores:',software).start()
            Institucion_financiadora = software[inicio:fin].strip().capitalize()
        except: 
            Institucion_financiadora = 'NA'

        info = {'titulo':Titulo, 'tipo':Tipo, 'autores':Autores, 'anio':year, 'disponibilidad':Disponibilidad, 'sitio_web':Sitio_web, 'nombre_comercial': Nombre_comercial, 
                'nombre_proyecto': Nombre_proyecto, 'institucion_financiadora': Institucion_financiadora, 'grupo':grupo}   
        softwares_formateados.append(info) 

    if len(softwares_formateados) == 0: 
        return softwares_formateados

    softwares_formateados = convertdf2json(convert2df(softwares_formateados).drop_duplicates('titulo'))   
    return(softwares_formateados)

# ----------------------------------------------------------------------------------------------
# --------- MODULO PARA DESCARGA DE LA INFORMACION ---------------------------------------------
# ----------------------------------------------------------------------------------------------

# DESCARGA DATOS INVESTIGAGORES 
async def read_researcher_info(url : str): 
    info_integrante = {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            assert response.status == 200
            response   = await response.read()
            integrante = pd.read_html(response)
            
            try:
                info_integrante['nombre'] = integrante[1][1][integrante[1][0]== 'Nombre'].values[0].replace(u'\xa0',' ').title()
            except: 
                info_integrante['nombre'] = ''

            try: 
                info_integrante['nombre_citaciones'] = integrante[1][1][integrante[1][0]== 'Nombre en citaciones'].values[0].replace(u'\xa0',' ').title()
            except: 
                info_integrante['nombre_citaciones'] = ''
            
            try: 
                info_integrante['categoria'] = integrante[1][1][integrante[1][0]== 'Categoría'].values[0]
            except: 
                info_integrante['categoria'] = ''
            
            try: 
                info_integrante['nacionalidad'] = integrante[1][1][integrante[1][0]== 'Nacionalidad'].values[0]
            except: 
                info_integrante['nacionalidad'] = ''

            try: 
                info_integrante['sexo'] = integrante[1][1][integrante[1][0]== 'Sexo'].values[0]
            except: 
                info_integrante['sexo'] = ''
            
            info_integrante['link']  = url
    return(info_integrante)


# DESCARGA INFORMACION PERFIL DE LOS GRUPOS 
async def read_group_profile(url : str, grupo : str):
    parametros_perfil = {'Perfil_integrantes':4, 'Colaboracion':6, 'Trayectoria_permanencia_estabilidad':8, 'Estabilidad_produccion':10,
                         'Generacion_nuevo_conocimiento' : 12, 'Desarrollo_tecnologico' : 14, 'Apropiacion_social_conocimiento':16, 
                         'formacion_recurso_humano':18}
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            assert response.status == 200
            html             = await response.text()
            gruplac_perfiles = pd.read_html(html)
            perfil = {}
            for key, value in parametros_perfil.items(): 
                try:
                    perfil[key] = gruplac_perfiles[value].to_dict()
                except: 
                    perfil[key] = np.NAN
                perfil['grupo']  = grupo
    return perfil 

# DESCARGA INFORMACION DE LOS GRUPOS      
async def read_group_info(url : str): 
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            assert response.status == 200

            # LECTURA INFORMACION BASICA 
            response       = await response.read()
            info           = pd.read_html(response)
            items_list     = [df.iloc[0,0] for df in info]
            soup           = BeautifulSoup(response, 'html.parser')
            nombre_instituciones = info[1][0].iloc[1:].str.replace('[0-9].-', '', regex=True).str.cat(sep=', ').strip()
            lineas_investigacion = info[3][0].iloc[1:].str.replace('[0-9].-', '', regex=True).str.cat(sep=', ').strip()
            val                  = soup.find_all('span', {'class' : 'celdaEncabezado'})
            nombre_grupo         = val[0].get_text()
            info_grupo           = info[0].drop(labels=0)
            informacion_basica                         = {key: val.strip().capitalize() for key, val in zip(info_grupo[0], info_grupo[1])}
            informacion_basica['Líder']                = informacion_basica['Líder'].title()
            informacion_basica['grupo']                = nombre_grupo.strip().capitalize()
            informacion_basica['instituciones']        = nombre_instituciones.strip().capitalize()
            informacion_basica['lineas_investigacion'] = lineas_investigacion.strip().capitalize()

            # INFORMACION DE INTEGRANTES
            idx_integrantes       = items_list.index('Integrantes del grupo')
            df_integrantes        = info[idx_integrantes]
            columnas              = df_integrantes.columns
            valores               = df_integrantes.iloc[1,:]
            new_columns           = {key : val for key, val in zip(columnas, valores)}
            df_integrantes.rename(columns=new_columns, inplace = True)
            df_integrantes.drop(index = [0,1], inplace=True)
            df_integrantes.Nombre = df_integrantes.Nombre.str.replace('[0-9].-','', regex=True).str.replace('[0-9]','', regex = True).str.title().str.strip()
            vinculacion           = df_integrantes['Inicio - Fin Vinculación'].str.split('-', expand=True).rename(columns={0:'inicio vinculacion', 1:'fin vinculacion'})
            df_integrantes        = pd.concat([df_integrantes, vinculacion], axis=1)
            df_integrantes.drop(columns = ['Inicio - Fin Vinculación'], inplace=True) 
            df_integrantes['grupo']     = nombre_grupo.strip().capitalize()

            link_integrantes = []
            for link in soup.find_all('a'):
                temp_link = link.get('href')
                if 'https://scienti.minciencias.gov.co/cvlac/visualizador/generarCurriculoCv.do?cod_rh' in temp_link:
                    link_integrantes.append(temp_link)

            df_integrantes['link'] = link_integrantes
            integrantes_json = convertdf2json(df_integrantes)
            
            # DESCARGA PRODUCCION CIENTIFICA 
            # Descarga y procesado de articulos
            idx_articulos  = items_list.index('Artículos publicados')
            info_articulos = info[idx_articulos].to_dict()
            articulos      = await  formato_articulos(articulos = list(info_articulos[1].values())[1:] , grupo = nombre_grupo)

            # Descarga y procesado de libros 
            idx_libros  = items_list.index('Libros de formación')
            info_libros = info[idx_libros].to_dict()
            libros      = await formato_libros(libros = list(info_libros[1].values())[1:], grupo = nombre_grupo)

            # Descarga y procesado de capitulos de libros 
            idx_capitulos  = items_list.index('Capítulos de libro publicados')
            info_capitulos = info[idx_capitulos].to_dict()
            capitulos      = await formato_libros(libros = list(info_capitulos[1].values())[1:], grupo = nombre_grupo)

            # Descarga y preproceso de trabajos de grado 
            idx_trabajos  = items_list.index('Trabajos dirigidos/turorías')
            info_trabajos = info[idx_trabajos].to_dict()
            trabajos      = await formato_trabajos_grado(trabajos_grado = list(info_trabajos[1].values())[1:], grupo = nombre_grupo)

            # Descarga y preproceso softwares 
            idx_softwares  = items_list.index('Softwares')
            info_softwares = info[idx_softwares].to_dict()
            softwares      = await formato_softwares(softwares = list(info_softwares[1].values())[1:], grupo = nombre_grupo)

            # Descarga y preproceso de innovaciones 
            idx_innovaciones  = items_list.index('Innovaciones en Procesos y Procedimientos')
            info_innovaciones = info[idx_innovaciones].to_dict()
            innovaciones      = await formato_innovaciones(innovaciones = list(info_innovaciones[1].values())[1:], grupo = nombre_grupo)

            produccion_cientifica = {'articulos' : articulos, 'libros' : libros, 'capitulos':capitulos, 
                                    'trabajos_grado': trabajos, 'softwares' :softwares, 'Innovaciones':innovaciones}

            return({'Informacion basica' : informacion_basica, 'Produccion cientifica':produccion_cientifica, 'integrantes':integrantes_json})


# DESCARGA DE DATOS DE LOS GRUPOS 
async def download_group_data(urls : list, skip : int = 10): 
    inicio         = 0
    fin            = inicio + skip 
    N              = len(urls)            
    results        = []
    errores        = []
    i = 0

    while inicio < N: 
        print('Datos de los grupos', i, inicio, fin, N, '----------------')
        try: 
            val = await asyncio.gather(*[read_group_info(url) for url in urls[inicio:fin]])
            results += val
        except: 
            try: 
                print('    Se va a intentar otra vez', inicio, fin)
                val = await asyncio.gather(*[read_group_info(url) for url in urls[inicio:fin]])
                results += val
            except:
                print('     No se pudo solucionar, descargar despues', inicio, fin)
                errores.append((inicio, fin))
        if (N - fin) < skip:
            skip = N - fin
        inicio = fin
        fin    = inicio + skip
    info = {'results': results, 'errors' : errores}
    return(info)



# DESCARGA DE DATOS ADICIONALES DE LOS INVESTIGADORES 
async def download_researcher_data(urls : list, skip : int = 10): 
    inicio         = 0
    fin            = inicio + skip 
    N              = len(urls)            
    results        = []
    errores        = []
    i = 0

    while inicio < N: 
        print('Datos de investigadores', i, inicio, fin, N, '----------------')
        try: 
            val = await asyncio.gather(*[read_researcher_info(url) for url in urls[inicio:fin]])
            results += val
        except: 
            try: 
                print('    Se va a intentar otra vez', inicio, fin)
                val = await asyncio.gather(*[read_researcher_info(url) for url in urls[inicio:fin]])
                results += val
            except:
                print('     No se pudo solucionar, descargar despues', inicio, fin)
                errores.append((inicio, fin))
        
        if (N - fin) < skip:
            skip = N - fin
        inicio = fin
        fin    = inicio + skip
    info = {'results':results, 'errors' : errores}
    return(info)

# DESCARGA DE PERFILES  DE LOS GRUPOS 
async def download_group_profile_data(urls : list, grupos: str, skip : int = 10): 
    inicio         = 0
    fin            = inicio + skip 
    N              = len(urls)            
    results        = []
    errores        = []
    i = 0

    while inicio < N: 
        print('Datos de perfil de los grupos', i, inicio, fin, N, '----------------')
        try: 
            val = await asyncio.gather(*[read_group_profile(url, grupo) for url, grupo in zip(urls[inicio:fin],grupos[inicio:fin])])
            results += val
        except: 
            try: 
                print('    Se va a intentar otra vez', inicio, fin)
                val = await asyncio.gather(*[read_group_profile(url, grupo) for url, grupo in zip(urls[inicio:fin],grupos[inicio:fin])])
                results += val
            except:
                print('     No se pudo solucionar, descargar despues', inicio, fin)
                errores.append((inicio, fin))
  
        if (N - fin) < skip:
            skip = N - fin
        inicio = fin
        fin    = inicio + skip
    
    info = {'results': await formato_perfiles_grupos(results), 'errors' : errores}
    return(info)

def relate_articles_publindex_scimago(articulos : list, scimago : pd.DataFrame, publindex_h : pd.DataFrame, publindex_n : pd.DataFrame): 
    articulos['anio'].fillna(0, inplace=True)
    articulos['anio'] = articulos['anio'].astype(int).astype(object)

    scimago.rename(columns={'ano':'anio'}, inplace=True)
    scimago.rename(columns={'ano':'anio'}, inplace=True)
    scimago.drop(scimago[scimago['ISSN']=='-'].index, inplace = True)
    scimago.rename(columns={'ISSN':'issn'}, inplace=True)

    publindex_h['ISSN'] = publindex_h['ISSN'].str.split(',')
    publindex_h = publindex_h.explode('ISSN')
    publindex_h.rename(columns={'ano':'anio'}, inplace=True)
    publindex_h['ISSN'] = publindex_h['ISSN'].str.replace('-','')
    publindex_h['anio'] = publindex_h['anio'].astype('object')
    publindex_h.rename(columns={'ISSN':'issn'}, inplace=True)

    publindex_n.rename(columns={'ano':'anio'}, inplace=True)
    publindex_n['ISSN'] = publindex_n['ISSN'].str.replace('-','')
    publindex_n['anio'] = publindex_n['anio'].astype('object')
    publindex_n['anio'] = publindex_n['anio'].fillna(0.0).astype(int)
    publindex_n['anio'] = publindex_n['anio'].astype(str)
    publindex_n.rename(columns={'ISSN':'issn'}, inplace=True)

    scimago_merge = pd.merge(scimago, publindex_h, on=['issn', 'anio'], how='outer')
    scimago_merge.drop(scimago_merge[scimago_merge['categoria_x']=='-'].index, inplace = True)
    scimago_merge = scimago_merge.assign(categoria = [b if pd.isnull(a) else a for a,b in zip(scimago_merge['categoria_x'], scimago_merge['categoria_y'])])
    scimago_merge.drop(['categoria_x','categoria_y','revista_y'], 1, inplace=True)
    scimago_merge.rename(columns={'revista_x':'revista'}, inplace=True)
    scimago_merge['categoria'] = scimago_merge['categoria'].replace(['A1','A2','B','C'],['Q1','Q2','Q3','Q4'])
    publindex_h['anio'] = publindex_h['anio'].fillna(0).astype(int).astype(object)
    scimago_merge['anio'] = scimago_merge['anio'].fillna(0).astype(int).astype(object)

    merged = pd.merge(articulos, scimago_merge, how='left', on=['issn','anio'])
    merge2 = pd.merge(merged, publindex_h, how='left', on=['issn','anio'])
    merge = pd.merge(merge2,publindex_n, how='left', on=['issn','anio'])
    merge = merge.drop_duplicates()
    merge['anio'] = merge['anio'].replace('NA',0)
    merge['anio'] = merge['anio'].fillna(0.0).astype(int)
    merge = merge.assign(categoria_y = [b if pd.isnull(a) else a for a,b in zip(merge['categoria_y'], merge['categoria'])])
    merge = merge.assign(revista_y = [b if pd.isnull(a) else a for a,b in zip(merge['revista_y'], merge['revista_x'])])
    merge = merge.assign(revista_y = [b if pd.isnull(a) else a for a,b in zip(merge['revista_y'], merge['revista'])])
    merge.drop(['revista_x','revista','categoria'], 1, inplace=True)
    merge = merge.assign(categoria_y = ['Sin información' if a<2003 else b for a,b in zip(merge['anio'], merge['categoria_y'])])
    merge = merge.assign(categoria_x = ['Sin información' if a<1999 else b for a,b in zip(merge['anio'], merge['categoria_x'])])
    merge = merge.assign(revista_y = ['Sin información' if pd.isnull(a) else a for a in merge['revista_y']])
    merge = merge.assign(categoria_x = ['Sin información' if a=='Sin información' else b for a,b in zip(merge['revista_y'], merge['categoria_x'])])
    merge = merge.assign(categoria_y = ['Sin información' if a=='Sin información' else b for a,b in zip(merge['revista_y'], merge['categoria_y'])])
    merge = merge.assign(categoria_x = ['Sin categoria' if pd.isnull(a) else a for a in merge['categoria_x']])
    merge = merge.assign(categoria_y = ['Sin categoria' if pd.isnull(a) else a for a in merge['categoria_y']])
    merge.rename(columns={'revista_y':'Revista', 'categoria_x':'SJR_Q', 'categoria_y':'Publindex'}, inplace=True)
    articulos_scimago_publindex = merge

    return(articulos_scimago_publindex)

# ------------------------------------------------------------------------------------------------------
# -------- EXTRACCION DE LOS DATOS DEL SIB -------------------------------------------------------------    
# ------------------------------------------------------------------------------------------------------


# Funcion para descarcar los registros del SiB relacionados con biologia 
def importar_datos_SiB():
    """Funcion para descargar los registros de conjuntos de datos del SiB
    Args:
       No requiere argumentos 
    Returns:
        Resultados: List con los registros de la base de datos de SiB 
    """
    offset = 0
    datos  = []
    fin    = False
    i      = 0 
    while not fin:
        parametros = {'q':'biologia', 'limit':1000, 'country':'CO','offset':offset}
        busqueda   = requests.get('https://api.gbif.org/v1/dataset/', params = parametros).json()
        offset    += len(busqueda['results'])
        datos.append(busqueda['results'])
        print(i, busqueda['endOfRecords'], offset)
        fin =  busqueda['endOfRecords']
        i += 1
    resultados = list(itertools.chain(*datos))
    return(resultados)

# Funcion para buscar los registros del SiB relacionados con biologia 
def buscar_datos_SiB():
    """Funcion para descargar los registros de conjuntos de datos del SiB
    Args:
       No requiere argumentos 
    Returns:
        Resultados: List con los registros de la base de datos de SiB 
    """
    offset = 0
    datos  = []
    fin    = False 
    i      = 0 
    while not fin:
        parametros = {'limit':1000, 'publishingCountry':'CO','offset':offset}
        busqueda   = requests.get('https://api.gbif.org/v1/dataset/search', params = parametros).json()
        offset    += len(busqueda['results'])
        datos.append(busqueda['results'])
        print(i, busqueda['endOfRecords'], offset)
        fin =  busqueda['endOfRecords']
        i += 1
    resultados = list(itertools.chain(*datos))
    return(resultados)

def download_SiB_data(): 
    """Funcion para descargar los datos del SiB 
        Args:
            No requiere argumentos 
        Returns:
            data_SiB: Datos descargados de la pagina del SiB 
    """
    resultados  = importar_datos_SiB()
    resultados2 = buscar_datos_SiB()
    df          = convert2df(resultados)
    df2         = convert2df(resultados2)
    columnas    = list(set(df2.columns).difference(set(df.columns))) 
    columnas.append('key')
    data_SIB_merged = df.merge(df2[columnas], left_on='key', right_on='key', suffixes=('_1','_2'), how = 'outer')
    data_SiB        = convertdf2json(data_SIB_merged)
    return(data_SiB)

class minciencias_data(AsyncClass): 
    """Clase para descarga de los datos de minciencias 

    Parameters:
        group_urls (list): Lista con las direcciones web de los grupos para descarga de la informacion
    
    Attributes: 
        group_data (dict): Diccionario con la toda la informacion de los grupos
        info_basica (List): Lista con informacion basica de los grupos
        articulos (List) : Lista con la informacion de los articulos 
        libros (List) : Lista con la informacion de los libros 
        capitulos (List) : Lista con la informacion de los capitulos de libros  
        trabajos_grado (List) : Lista con la informacion de los capitulos de trabajos de grado   
        softwares (List) : Lista con la informacion de los  softwares 
        Innovaciones (List) : Lista con la informacion de las innovaciones   
        integrantes (List): Lista con la informacion de los integrantes de los grupos 
        profile_data (List) : Lista con la informacion de los perfiles de los grupos 
        articulos_publindex_scimago (List) : Lista con los articulos relacionados con publindex 
    """

    def __init__(self, group_urls: list):
        self.urls         = group_urls
        self.group_data   = None
        self.info_basica  = None
        self.articulos    = None
        self.libros       = None 
        self.capitulos    = None 
        self.softwares    = None
        self.Innovaciones = None 
        self.integrantes  = None 
        self.profile_data = None 
        self.articulos_publindex_scimago = None 

    async def get_group_data(self): 
        """ Funcion para obtener los datos de los grupos 
        """
        self.group_data      = await download_group_data(self.urls)
        self.info_basica     = [info['Informacion basica'] for info in self.group_data['results']]
        self.articulos       = list(itertools.chain(*[info['Produccion cientifica']['articulos'] for info in self.group_data['results']]))
        self.libros          = list(itertools.chain(*[info['Produccion cientifica']['libros'] for info in self.group_data['results']]))
        self.capitulos       = list(itertools.chain(*[info['Produccion cientifica']['capitulos'] for info in self.group_data['results']]))
        self.trabajos_grado  = list(itertools.chain(*[info['Produccion cientifica']['trabajos_grado'] for info in self.group_data['results']]))
        self.softwares       = list(itertools.chain(*[info['Produccion cientifica']['softwares'] for info in self.group_data['results']]))
        self.Innovaciones    = list(itertools.chain(*[info['Produccion cientifica']['softwares'] for info in self.group_data['results']]))
        self.integrantes     = list(itertools.chain(*[info['integrantes'] for info in self.group_data['results']]))
    
    async def get_researcher_data(self): 
        """Funcion para obtener los datos de los integrantes de los grupos 
        """
        integrantes_df        = convert2df(self.integrantes)
        links_integrantes     = integrantes_df.link
        integrantes           = await download_researcher_data(links_integrantes)
        integrantes_adicional = convert2df(integrantes['results'])
        integrantes_df_merged = pd.merge(left=integrantes_df, right=integrantes_adicional, how='left', left_on='Nombre', right_on='nombre')[['Nombre', 'Vinculación', 'Horas dedicación', 'inicio vinculacion',
                                    'fin vinculacion', 'link_x', 'nombre_citaciones', 'categoria',
                                    'nacionalidad', 'sexo', 'grupo']]
        integrantes_df_merged.rename(columns = {'link_x':'link'}, inplace = True)
        self.integrantes = [row.to_dict() for idx, row in integrantes_df_merged.iterrows()]

    async def get_group_profile(self, profile_urls: list, group_names : list): 
        """Funcion para obtener los datos de los perfiles de los grupos 

        Args:
            profile_urls (list): Lista con las direcciones web de los perfiles de los grupos de investigacion 
            group_names (list): Nombre de los grupos de investigacion 
        """
        self.profile_data = await download_group_profile_data(profile_urls, group_names)
    
    def relate_publindex_scimago(self, scimago : pd.DataFrame, publindex_h : pd.DataFrame, publindex_n : pd.DataFrame):
        """Funcion para relacionar los articulos descargados con la clasificacion de publindex 

        Args:
            scimago (pd.DataFrame): Data frame con la informacion de scimago 
            publindex_h (pd.DataFrame): Data frame con la informacion de publindex homologados 
            publindex_n (pd.DataFrame): Data frame con la informacion de publindex nacionales 
        """
        articulos = convert2df(self.articulos)
        self.articulos_publindex_scimago = convertdf2json(relate_articles_publindex_scimago(articulos, scimago, publindex_h, publindex_n))
        return self.articulos_publindex_scimago
    
async def main(group_url : list, download_researcher: bool =False, profiles_groups : list = None, name_groups : list = None,):
    """Funcion para descargar los datos de minciencias 
    Args:
        group_url (list): Lista con las direcciones web de los grupos 
        profiles_groups (list, optional): Lista con las direcciones web de los perfiles de los grupos. Defaults to None.
        name_groups (list, optional): Lista con los nombres de los grupos para descargar perfiles. Defaults to None.
        download_researcher (bool, optional): _description_. Defaults to True.
    
    Returns: 
        minciencias_class (class): clase con los atributos de los datos de minciencias 
    """
    minciencias_class = await minciencias_data(group_url)
    # Se descargan los datos de los grupos de investigacion de minciencias 
    await minciencias_class.get_group_data()

    # Se descarga la informacion adicional de los integrantes de los grupos 
    if download_researcher:
      await minciencias_class.get_researcher_data()
    
     # Si se brindan los nombres de los grupos y las direcciones web de los perfiles de los grupos se descargan estos datos
    if (profiles_groups is not None) and (len(profiles_groups) == len(name_groups)) and (name_groups is not None): 
        await minciencias_class.get_group_profile(profiles_groups, name_groups)
        
    return(minciencias_class)