// Conforming curved dielectric interface, square far field for Cartesian PML.
SetFactory("OpenCASCADE");
If (!Exists(a)) a=.5; EndIf
If (!Exists(L)) L=2.5; EndIf
If (!Exists(h)) h=.3; EndIf
Rectangle(1)={-L,-L,0,2*L,2*L};
Disk(2)={0,0,0,a,a};
parts()=BooleanFragments{Surface{1};Delete;}{Surface{2};Delete;};
cylinder()=Surface In BoundingBox{-a-1e-6,-a-1e-6,-1e-6,a+1e-6,a+1e-6,1e-6};
air()=parts(); air()-=cylinder();
outer()=CombinedBoundary{Surface{parts()};};
interface()=Curve In BoundingBox{-a-1e-6,-a-1e-6,-1e-6,a+1e-6,a+1e-6,1e-6};
Physical Surface("air",1)={air()};
Physical Surface("dielectric",2)={cylinder()};
Physical Curve("outer",1)={outer()};
Physical Curve("interface",2)={interface()};
Mesh.MeshSizeMin=h; Mesh.MeshSizeMax=h;
Mesh.MeshSizeFromCurvature=24;
Mesh.SecondOrderIncomplete=0;
