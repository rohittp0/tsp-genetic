import cv2

from main import variables, draw_map, draw_solution
from problem import create_distance_matrix


def main():
    sea_map, ports, lands, green = variables()
    sea_map = draw_map(sea_map, ports, lands, green)

    a, b = create_distance_matrix([ports[6], ports[0]], green, lands)

    sea_map = draw_solution(sea_map, [1, 0], [ports[6], ports[0]], b)

    cv2.imshow('map', sea_map)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
