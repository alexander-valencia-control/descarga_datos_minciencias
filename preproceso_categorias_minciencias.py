import aiohttp
import asyncio
import nest_asyncio
nest_asyncio.apply()
import sys 
import json
import pandas as pd 
import re 
import numpy as np 
import os

# Funciones para descarga asincrona 
def convert2df(list_json): 
    keys    = list_json[0].keys()
    df_dict = {key:[] for key in keys}
    for val in list_json: 
        for key in keys: 
            df_dict[key].append(val[key])
    return pd.DataFrame(df_dict)
        
async def formato_articulos(articulos, grupo): 
    # Extraccion de titulo
    articulos_fortmateados = []
    for articulo in articulos: 
        try:
            inicio_titulo = re.search(':',articulo).end()
            fin_titulo    = re.search('  ',articulo).start()
            Titulo = articulo[inicio_titulo:fin_titulo].strip()
        except:
            Titulo = 'NA'
        # Extraccion del DOI
        try:
            DOI = re.findall('(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)',articulo)[0]
        except:
            DOI = 'NA'
        # Extraccion del ISSN 
        try: 
            ISSN = re.sub('-','',re.findall('\d{4}-\d{3}[\d|X]',articulo)[0])
        except:
            ISSN = 'NA'
        # Extraccion de autores 
        try: 
            inicio_autores = re.search('Autores:',articulo).end()
            Autores = articulo[inicio_autores:].strip()
        except:
            Autores = 'NA'
            
        # Extraccion del año 
        try:
            Year = re.sub(',','',re.findall(' \d{4} ',articulo)[0],).strip()
        except: 
            Year = 'NA'
            
        # Extraccion del tipo de articulo 
        try:
            inicio = re.search('-',articulo).end()
            fin    = re.search(':',articulo).start()
            Tipo_articulo = articulo[inicio:fin].strip()
        except:
            Tipo_articulo = 'NA'
        info = {'Titulo':Titulo, 'DOI':DOI, 'ISSN':ISSN, 'Autores':Autores, 'Tipo_articulo':Tipo_articulo, 'year':Year, 'grupo':grupo}
        articulos_fortmateados.append(info)
    return(articulos_fortmateados)


# Funcion para extraer informacion de libros  
async def formato_libros(libros : str, grupo : str):

    libros_formateados = []
    for libro  in libros :
        # Extraccion del titulo 
        try:
            inicio_titulo = re.search(':',libro).end()
            fin_titulo = re.search('  ',libro).start()
            Titulo = libro[inicio_titulo:fin_titulo].strip()
        except: 
            Titulo = 'NA'
            
        # Extraccion de Autores
        try:
            inicio_autores = re.search('Autores:',libro).end()
            Autores = libro[inicio_autores:].strip()
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
            Tipo_libro = libro[inicio:fin].strip()
        except:
            Tipo_libro = 'NA'
        info = {'Titulo':Titulo,'Autores':Autores, 'Tipo_libro':Tipo_libro, 'year':Year, 'ISBN': ISBN, 'Grupo':grupo}   
        libros_formateados.append(info)
            
    return(libros_formateados)


async def formato_innovaciones(innovaciones, grupo):
    
    innovaciones_formateadas = []

    for innovacion in innovaciones: 
        # Extraccion de tipo 
        try: 
            inicio = re.search('.-', innovacion).end()
            fin    = re.search(' : ', innovacion).start()
            tipo = innovacion[inicio:fin].strip()
        except: 
            tipo.append('NA')
        # Extraccion de titulo
        try: 
            inicio = re.search(' : ', innovacion).end()
            fin    = re.search('[1,2]\d{3}', innovacion).start()
            titulo = innovacion[inicio:fin].replace(',','').strip()
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
            Disponibilidad = innovacion[inicio:fin].replace(',','').strip()
        except: 
            Disponibilidad = 'NA'
        # Extraccion de institucion  
        try: 
            inicio = re.search('Institución financiadora:', innovacion).end()
            fin    = re.search('Autores:', innovacion).start()
            Institucion = innovacion[inicio:fin].replace(',','').strip()
        except: 
            Institucion = 'NA'
        # Extraccion de autores   
        try: 
            inicio = re.search('Autores:', innovacion).end()
            Autores = (innovacion[inicio:].replace(',','').strip())
        except: 
            Autores = 'NA'

        info = {'tipo':tipo, 'titulo':titulo, 'ano':ano, 'Autores':Autores,'Disponibilidad':Disponibilidad, 'Nombre_grupo': grupo, 'Institucion':Institucion, 'Autores':Autores}    
        innovaciones_formateadas.append(info)       
    return(innovaciones_formateadas)



async def formato_trabajos_grado(trabajos_grado, grupo): 
    trabajos_grado_formateados = []
    for trabajo in trabajos_grado: 
        try: 
            inicio_titulo = re.search(' : ',trabajo).end()
            fin_titulo = re.search('  ', trabajo).start()
            Titulo = trabajo[inicio_titulo:fin_titulo].strip()
        except: 
            Titulo = 'NA'

        try: 
            inicio_autor = re.search('Nombre del estudiante:',trabajo).end()
            fin_autor = re.search(',  Programa', trabajo).start()
            Autores = trabajo[inicio_autor:fin_autor].strip()
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
            Tipo_producto = trabajo[inicio:fin].strip()
        except: 
            Tipo_producto = 'NA'
            
        try: 
            inicio = re.search('Cotutor',trabajo).end()
            tutor = trabajo[inicio+5:].strip()
        except:
            tutor = 'NA'
            
        try: 
            inicio = re.search('Institución:',trabajo).end()
            fin    = re.search('Cotutor',trabajo).start()
            institucion = trabajo[inicio:fin-10].strip()
        except: 
            institucion = 'NA'
            
        try: 
            inicio = re.search('Programa académico:',trabajo).end()
            fin    = re.search('Número de páginas:',trabajo).start()
            programa_academico = trabajo[inicio:fin].strip()
        except: 
            programa_academico = 'NA'
            
        try: 
            inicio = re.search('Número de páginas:',trabajo).end()
            fin    = re.search(', Valoración:',trabajo).start()
            Numero_paginas = trabajo[inicio:fin].strip() 
        except: 
            Numero_paginas = 'NA'
        info = {'Titulo':Titulo,'Autores':Autores, 'Desde':Desde, 'Hasta':Hasta, 'Tipo_producto': Tipo_producto, 'Grupo':grupo,
                'tutor':tutor, 'Tipo_producto':Tipo_producto, 'institucion':institucion, 'programa_academico':programa_academico, 
                'Numero_paginas':Numero_paginas}     
        trabajos_grado_formateados.append(info)      
        
    return(trabajos_grado_formateados)

async def formato_softwares(softwares, grupo): 
    softwares_formateados = []
    for software in softwares: 
        # TIpo software 
        try: 
            inicio = re.search('\d{1}.-',software).end() 
            fin    = re.search(' : ', software).start()
            Tipo   = software[inicio:fin].strip()
        except: 
            Tipo = 'NA'

        # Titulo 
        try: 
            inicio_titulo = re.search(' : ',software).end()
            fin_titulo = re.search('  ', software).start()
            Titulo = software[inicio_titulo:fin_titulo].strip()
        except: 
            Titulo = 'NA'
        # Autores 
        try: 
            inicio_autor = re.search('Autores:',software).end()
            Autores = software[inicio_autor:].strip()
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
            Disponibilidad = software[inicio:fin].strip().replace(',','')
        except: 
            Disponibilidad = 'NA'
        
        # Sitio web 
        try: 
            inicio = re.search('Sitio web:',software).end()
            fin    = re.search('Nombre comercial:',software).start()
            Sitio_web = software[inicio:fin].strip()
        except:
            Sitio_web = 'NA'
            
        try: 
            inicio = re.search('Nombre comercial:', software).end()
            fin    = re.search('Nombre del proyecto:',software).start()
            Nombre_comercial = software[inicio:fin-10].strip()
        except: 
            Nombre_comercial = 'NA'
            
        try: 
            inicio = re.search('Nombre del proyecto:',software).end()
            fin    = re.search('Institución financiadora:',software).start()
            Nombre_proyecto = software[inicio:fin].strip()
        except: 
            Nombre_proyecto = 'NA'
            
        try: 
            inicio = re.search('Institución financiadora:',software).end()
            fin    = re.search('Autores:',software).start()
            Institucion_financiadora = software[inicio:fin].strip() 
        except: 
            Institucion_financiadora = 'NA'

        info = {'Titulo':Titulo, 'Tipo':Tipo, 'Autores':Autores, 'Year':year, 'Disponibilidad':Disponibilidad, 'Sitio_web':Sitio_web, 'Nombre_comercial': Nombre_comercial, 
                'Nombre_proyecto': Nombre_proyecto, 'Institucion_financiadora': Institucion_financiadora, 'Grupo':grupo}   
        softwares_formateados.append(info)      
        
    return(softwares_formateados)