[xg, yg] = meshgrid(-3:0.25:3);
xg = xg(:);
yg = yg(:);

t = (pi/24:pi/24:2*pi)';
x = cos(t);
y = sin(t);
circShp = alphaShape(x,y,2);
in = inShape(circShp,xg,yg);
xg = [xg(~in); cos(t)];
yg = [yg(~in); sin(t)];

zg = ones(numel(xg),1);
xg = repmat(xg,5,1);
yg = repmat(yg,5,1);
zg = zg*(0:.25:1);
zg = zg(:);
shp = alphaShape(xg,yg,zg);

[elements,nodes] = boundaryFacets(shp);

nodes = nodes';
elements = elements';

model = createpde();
geometryFromMesh(model,nodes,elements);

pdegplot(model,'FaceLabels','on','FaceAlpha',0.5)

generateMesh(model);








