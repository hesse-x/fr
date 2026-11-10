// Non-orthogonal, unstructured quadrilaterals (Gmsh may retain a few triangles).
Point(1)={0,0,0,.2}; Point(2)={1,0,0,.2};
Point(3)={1.15,.9,0,.2}; Point(4)={.15,1.1,0,.2};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Curve Loop(1)={1,2,3,4}; Plane Surface(1)={1};
Mesh.Algorithm=6;
Recombine Surface{1};
Mesh.SecondOrderIncomplete=0;
Physical Surface("domain",1)={1};
Physical Curve("boundary",1)={1,2,3,4};
