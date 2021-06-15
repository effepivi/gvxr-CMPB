// Matrix

color("red")

//difference() {
    scale([15, 70, 70])
        cube(1, center = true);
//    make_spheres([2, 3.5, 5], 50, 5, 40, 3, -1);
//}

/*
color("blue")
    make_spheres([2, 3.5, 5], 50, 5, 40, 3, 4);
*/

module make_column_of(sphere_radius, height, count)
{
    step = height / (count - 1);
    for (a = [0 : count - 1]) {
        offset = -height / 2 + step * a ;
        translate([0, offset, 0])
            sphere(sphere_radius[a], $fa=5, $fs=0.1);
    }
}

module make_row_of(radius, count, id)
{
    step = radius / (count - 1);
    for (a = [0 : count - 1]) {
        if (id == -1 || id == a) {
            offset = -radius / 2 + step * a ;
            translate([0, 0, offset])
                children();
        }
    }
}

module make_spheres(sphere_radius, ring_radius, ring_count, column_height, column_count, id = -1)
{
    make_row_of(radius = ring_radius, count = ring_count, id = id)
        make_column_of(sphere_radius, height = column_height, count = column_count);
}
