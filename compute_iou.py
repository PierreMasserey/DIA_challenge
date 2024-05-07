import pandas as pd
import os
from shapely.geometry import Polygon
import re

def create_polygons(csv):
    polygons = {}
    current_id = None
    current_points = []

    df = pd.read_csv(os.path.realpath(csv))

    for idx, row in df.iterrows():
        color = str(row['Color'])  # Convertir en chaîne de caractères
        if current_id is None:
            current_id = color
        if color != current_id:
            if current_id not in polygons:
                polygons[current_id] = []
            if len(current_points) >= 3:
                polygons[current_id].append(current_points)
            current_points = []
            current_id = color
        current_points.append((row['X'], row['Y']))

    if current_id not in polygons:
        polygons[current_id] = []
    if len(current_points) >= 3:
        polygons[current_id].append(current_points)

    return polygons


def calculate_iou(poly1, poly2):
    return Polygon(poly1).intersection(Polygon(poly2)).area / Polygon(poly1).union(Polygon(poly2)).area

def match_polygons(polygons1, polygons2):
    matched_polygons = {}

    # Créer des groupes de polygones pour chaque couleur dans chaque jeu de données
    groups1 = {}
    groups2 = {}

    for id1, points_list1 in polygons1.items():
        color = id1.split('_')[0] if id1 is not None else None  # Extraire la couleur de l'ID
        if color not in groups1:
            groups1[color] = []
        groups1[color].extend([(id1, points) for points in points_list1])

    for id2, points_list2 in polygons2.items():
        color = id2.split('_')[0] if id2 is not None else None  # Extraire la couleur de l'ID
        if color not in groups2:
            groups2[color] = []
        groups2[color].extend([(id2, points) for points in points_list2])

    # Comparer les polygones de chaque couleur
    for color, points_list2 in groups2.items():
        if color not in groups1:
            print(f"Aucun polygone correspondant trouvé pour la couleur {color} dans le premier fichier.")
            continue

        matched_polygons[color] = []  # Initialiser la liste des correspondances pour cette couleur

        matched_ids = set()  # Garder une trace des IDs du fichier 1 qui ont des correspondances dans le fichier 2

        for id2, points2 in points_list2:
            max_iou = -1
            matching_polygon = None
            matching_id = None
            for id1, points1 in groups1[color]:
                iou = calculate_iou(points1, points2)
                if iou > max_iou:
                    max_iou = iou
                    matching_polygon = points1
                    matching_id = id1  # Garder l'ID du premier fichier pour le polygone correspondant
            if matching_id not in matched_ids:
                matched_ids.add(matching_id)
                matched_polygons[color].append({'id': matching_id, 'polygon': matching_polygon})

    return matched_polygons




def compute_iou(csv1, csv2):
    polygons1 = create_polygons(csv1)
    polygons2 = create_polygons(csv2)
    matched_polygons = match_polygons(polygons1, polygons2)

    # Compter le nombre d'IDs dans chaque fichier
    num_ids_csv1 = len(polygons1)
    num_ids_csv2 = len(polygons2)

    # Compter le nombre d'IDs dans le fichier 2 ayant un ID correspondant dans le fichier 1
    num_matched_ids = sum(len(matches) for matches in matched_polygons.values())

    # Trouver les IDs dans le fichier 1 sans correspondance dans le fichier 2
    unmatched_ids_csv1 = [id for id in polygons1 if id not in matched_polygons]

    # Trouver les IDs dans le fichier 2 sans correspondance dans le fichier 1
    unmatched_ids_csv2 = [id for id in polygons2 if id.split('_')[0] not in matched_polygons]

    # Ajouter les IDs sans correspondance dans le fichier 2 uniquement si le fichier 1 est différent du fichier 2
    if csv1 != csv2:
        num_ids_csv2 += len(unmatched_ids_csv1)

    print(f"Nombre d'IDs dans le fichier 1 : {num_ids_csv1}")
    print(f"Nombre d'IDs dans le fichier 2 : {num_ids_csv2}")
    print(f"Nombre d'IDs du fichier 2 avec un ID correspondant dans le fichier 1 : {num_matched_ids+1}")

    # Afficher les correspondances
    for color, matches in matched_polygons.items():
        for matched_polygon in matches:
            print(f"Polygone ID {matched_polygon['id']} du premier fichier correspond au polygone ID {color} du deuxième fichier.")

    # Afficher les IDs sans correspondance dans le fichier 2
    for id in unmatched_ids_csv2:
        print(f"Aucun polygone correspondant trouvé pour l'ID {id} dans le deuxième fichier.")

#compute_iou("test.csv", "test2.csv")



            
