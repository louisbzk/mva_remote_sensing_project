
%% test image
I = im2double(imread('img.png'));
I = log(I+1/255);
I = I(10:end-10,20:end-10,1);
newimage(I);
Im = I;

%% estimation d'un profil
patch_half_size = 10;
profil_ext = ceil(sqrt(2)*patch_half_size);

[X,Y] = meshgrid(-patch_half_size:patch_half_size,-patch_half_size:patch_half_size);

theta = 20;
I = [1:numel(X); 1:numel(X)];
I = I(:);
J = 0*I;
V = J;
k = 0;
for i=1:size(X,1)
    for j=1:size(X,2)
        p = X(i,j)*cos(theta)+Y(i,j)*sin(theta);
        J(2*k+1) = floor(p+profil_ext+1);
        J(2*k+2) = ceil(p+profil_ext+1);
        V(2*k+1) = abs(p-ceil(p));
        V(2*k+2) = abs(p-floor(p));
        k = k+1;
    end
end
M = sparse(I,J,V,numel(X),2*profil_ext+1);

%% application d'un profil
profil = rand(2*profil_ext+1,1);
patch = reshape(M*profil,[2*patch_half_size+1,2*patch_half_size+1]);
newimage(patch);

%% idem mais avec la condition de symétrie en plus:
patch_half_size = 4;
profil_ext = ceil(sqrt(2)*(patch_half_size+1));

[X,Y] = meshgrid(-patch_half_size:patch_half_size,-patch_half_size:patch_half_size);

theta = 2*pi*(60/360);
I = [1:numel(X); 1:numel(X)];
I = I(:);
J = 0*I;
V = J;
k = 0;
for i=1:size(X,1)
    for j=1:size(X,2)
        p = abs(X(i,j)*cos(theta)+Y(i,j)*sin(theta));
        J(2*k+1) = floor(p+1);
        J(2*k+2) = ceil(p+1);
        V(2*k+1) = abs(p-ceil(p));
        V(2*k+2) = 1-V(2*k+1);%abs(p-floor(p));
        k = k+1;
    end
end
M = sparse(I,J,V,numel(X),profil_ext);

%% application d'un profil
profil = rand(profil_ext,1);
patch = reshape(M*profil,[2*patch_half_size+1,2*patch_half_size+1]);
newimage(patch);

%% application d'un profil avec contrainte >= centre
profil = rand(profil_ext,1);
profil = max(profil,profil(1));
patch = reshape(M*profil,[2*patch_half_size+1,2*patch_half_size+1]);
newimage(patch);


%% estimation de la carte de détection de structure linéique selon l'orientation theta:
mu = 1E-6;
A = M*(M'*M+mu*eye(profil_ext))^(-1)*(M');
[U,D] = eig(A'*A-2*A);
for k=1:size(U,1)
    if abs(D(k,k))>1E-4*abs(D(1,1))
        newimage(reshape(U(:,k),[patch_half_size*2+1 patch_half_size*2+1]));
        abs(D(k,k))
    end
end

%
C = 0*Im;
for k=1:size(U,1)
    if abs(D(k,k))>1E-4*abs(D(1,1))
        C = C+ abs(D(k,k))^2*imfilter(Im,reshape(U(:,k),[patch_half_size*2+1 patch_half_size*2+1])).^2;
    end
end
newimage(Im);
newimage(C);


%% Max sur plusieurs orientations:
patch_half_size = 25;
profil_ext = ceil(sqrt(2)*(patch_half_size+1));
Cmax = 0*Im;
figure;
for theta = 2*pi*(0:5:160)/360

    [X,Y] = meshgrid(-patch_half_size:patch_half_size,-patch_half_size:patch_half_size);

    %theta = 2*pi*(10/360);
    I = [1:numel(X); 1:numel(X)];
    I = I(:);
    J = 0*I;
    V = J;
    k = 0;
    for i=1:size(X,1)
        for j=1:size(X,2)
            p = abs(X(i,j)*cos(theta)+Y(i,j)*sin(theta));
            J(2*k+1) = floor(p+1);
            J(2*k+2) = ceil(p+1);
            V(2*k+1) = abs(p-ceil(p));
            V(2*k+2) = 1-V(2*k+1);%abs(p-floor(p));
            k = k+1;
        end
    end
    M = sparse(I,J,V,numel(X),profil_ext);
    %
    A = M*(M'*M+mu*eye(profil_ext))^(-1)*(M');
    [U,D] = eig(A'*A-2*A);

    C2 = 0*Im;
    for k=1:size(U,1)
        if abs(D(k,k))>1E-4*abs(D(1,1))
            C2 = C2 + abs(D(k,k))^2*imfilter(Im,reshape(U(:,k),[patch_half_size*2+1 patch_half_size*2+1])).^2;
        end
    end
    Cmax = max(Cmax,C2);
    imagesc(Cmax); title(sprintf('theta=%d',round(360*theta/(2*pi)))); drawnow;
end
newimage(Cmax);



%% Autre approche:
patch_half_size = 10;
profil_ext = ceil(sqrt(2)*(patch_half_size+1));
Cmax = 0*Im;
mu = 1E-2;
figure;
angular_step = 1;
for theta = 2*pi*(0:angular_step:180-angular_step)/360

    [X,Y] = meshgrid(-patch_half_size:patch_half_size,-patch_half_size:patch_half_size);

    %theta = 2*pi*(10/360);
    I = [1:numel(X); 1:numel(X)];
    I = I(:);
    J = 0*I;
    V = J;
    k = 0;
    for i=1:size(X,1)
        for j=1:size(X,2)
            p = abs(X(i,j)*cos(theta)+Y(i,j)*sin(theta));
            J(2*k+1) = floor(p+1);
            J(2*k+2) = ceil(p+1);
            V(2*k+1) = abs(p-ceil(p));
            V(2*k+2) = 1-V(2*k+1);%abs(p-floor(p));
            k = k+1;
        end
    end
    M = sparse(I,J,V,numel(X),profil_ext);

    %
    % 1/ on calcule le profil:
    P = (M'*M+mu*eye(profil_ext))^(-1)*(M');
    Profiles = zeros(size(Im,1),size(Im,2),size(P,1));
    for i=1:size(P,1)
        Profiles(:,:,i) = imfilter(Im,reshape(P(i,:),[patch_half_size*2+1 patch_half_size*2+1]));
    end

    % 2/ on interdit les valeurs inférieures à celle au centre du profil
    Profiles = max(Profiles,Profiles(:,:,1));

    % 3/ on calcule l'énergie du profil
    [U,S,V] = svd(full(M),'econ');
    Proj = 0*Profiles;
    for i=1:size(V,1)
        Proj(:,:,i) = S(i,i)*sum(Profiles.*reshape(V(:,i),[1,1,size(V,1)]),3);
    end
    E = sum(Proj.^2,3);

    Cmax = max(Cmax,E);
    imagesc(Cmax); title(sprintf('theta=%d',round(360*theta/(2*pi)))); drawnow;
end
newimage(Cmax);

%% Même calcul sur différentes échelles:
I = im2double(imread('img.png'));
I = log(I+1/255);
I = I(10:end-10,20:end-10,1);
Im = I;
Im_orig = Im;

MultiScaleResult = 0*Im_orig;

for scale = 6:-1:1
    Im = imresize(Im_orig,1/scale);
    patch_half_size = 4;
    profil_ext = ceil(sqrt(2)*(patch_half_size+1));
    Cmax = 0*Im;
    mu = 1E-2;
    figure;
    angular_step = 1;
    for theta = 2*pi*(0:angular_step:180-angular_step)/360

        [X,Y] = meshgrid(-patch_half_size:patch_half_size,-patch_half_size:patch_half_size);

        %theta = 2*pi*(10/360);
        I = [1:numel(X); 1:numel(X)];
        I = I(:);
        J = 0*I;
        V = J;
        k = 0;
        for i=1:size(X,1)
            for j=1:size(X,2)
                p = abs(X(i,j)*cos(theta)+Y(i,j)*sin(theta));
                J(2*k+1) = floor(p+1);
                J(2*k+2) = ceil(p+1);
                V(2*k+1) = abs(p-ceil(p));
                V(2*k+2) = 1-V(2*k+1);%abs(p-floor(p));
                k = k+1;
            end
        end
        M = sparse(I,J,V,numel(X),profil_ext);

        %
        % 1/ on calcule le profil:
        P = (M'*M+mu*eye(profil_ext))^(-1)*(M');
        Profiles = zeros(size(Im,1),size(Im,2),size(P,1));
        for i=1:size(P,1)
            Profiles(:,:,i) = imfilter(Im,reshape(P(i,:),[patch_half_size*2+1 patch_half_size*2+1]));
        end

        % 2/ on interdit les valeurs inférieures à celle au centre du profil
        Profiles = max(Profiles,Profiles(:,:,1));

        % 3/ on calcule l'énergie du profil
        [U,S,V] = svd(full(M),'econ');
        Proj = 0*Profiles;
        for i=1:size(V,1)
            Proj(:,:,i) = S(i,i)*sum(Profiles.*reshape(V(:,i),[1,1,size(V,1)]),3);
        end
        E = sum(Proj.^2,3);

        Cmax = max(Cmax,E);
    end
    MultiScaleResult = MultiScaleResult + imresize(Cmax,size(MultiScaleResult));
    imagesc(MultiScaleResult); title(sprintf('scale=%d',scale)); drawnow;
end


%% GLRT sur différentes échelles:
I = im2double(imread('img.png'));
I = log(I+1/255);
I = I(10:end-10,20:end-10,1);
Im = I;
Im_orig = Im;

MultiScaleResult = 0*Im_orig;

figure;
for scale = 6:-1:1
    Im = imresize(Im_orig,1/scale);
    patch_half_size = 4;%10;%10;%4;
    profil_ext = ceil(sqrt(2)*(patch_half_size+1));
    Cmin = 0*Im+inf;
    mu = 1E-2;
    angular_step = 1;
    for theta = 2*pi*(0:angular_step:180-angular_step)/360

        [X,Y] = meshgrid(-patch_half_size:patch_half_size,-patch_half_size:patch_half_size);

        %theta = 2*pi*(10/360);
        I = [1:numel(X); 1:numel(X)];
        I = I(:);
        J = 0*I;
        V = J;
        k = 0;
        for i=1:size(X,1)
            for j=1:size(X,2)
                p = abs(X(i,j)*cos(theta)+Y(i,j)*sin(theta));
                J(2*k+1) = floor(p+1);
                J(2*k+2) = ceil(p+1);
                V(2*k+1) = abs(p-ceil(p));
                V(2*k+2) = 1-V(2*k+1);%abs(p-floor(p));

                k = k+1;
            end
        end
        M = sparse(I,J,V,numel(X),profil_ext);

        %
        % 1/ on calcule le profil:
        P = (M'*M+mu*eye(profil_ext))^(-1)*(M');
        Profiles = zeros(size(Im,1),size(Im,2),size(P,1));
        for i=1:size(P,1)
            Profiles(:,:,i) = imfilter(Im,reshape(P(i,:),[patch_half_size*2+1 patch_half_size*2+1]));
        end

        % 2/ on interdit les valeurs inférieures à celle au centre du profil
        Profiles = max(Profiles,Profiles(:,:,1));

        % 3/ on calcule l'énergie des résidus
        E1 = imfilter(Im.^2,ones(patch_half_size*2+1));
        Mf = full(M);
        E2 = 0*E1;
        for i=1:size(M,2)
            E2 = E2-2*squeeze(Profiles(:,:,i)).*imfilter(Im,reshape(Mf(:,i),[patch_half_size*2+1 patch_half_size*2+1]));
        end
        [U,S,V] = svd(full(M),'econ');
        Proj = 0*Profiles;
        for i=1:size(V,1)
            Proj(:,:,i) = S(i,i)*sum(Profiles.*reshape(V(:,i),[1,1,size(V,1)]),3);
        end
        E3 = sum(Proj.^2,3);

        %E4 = 0*E3;%-imfilter((Im-imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2)).^2,ones(patch_half_size*2+1));
        %E4 = -imfilter((Im-imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2)).^2,ones(patch_half_size*2+1));
        E4 = -(imfilter(Im.^2,ones(patch_half_size*2+1))-(patch_half_size*2+1)^2*imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2).^2);


%         figure;
%         subplot(2,2,1); imagesc(E1);
%         subplot(2,2,2); imagesc(E2);
%         subplot(2,2,3); imagesc(E3);
%         subplot(2,2,4); imagesc(E1+E2+E3);
%         pause;

        Cmin = min(Cmin,(E1+E2+E3+E4));
        imagesc(Cmin); title(sprintf('theta=%d',round(360*theta/(2*pi)))); drawnow;
    end
    MultiScaleResult = MultiScaleResult + imresize(Cmin,size(MultiScaleResult));
    imagesc(MultiScaleResult); title(sprintf('scale=%d',scale)); drawnow;
end
%figure; imagesc(min(0,MultiScaleResult));
figure; imagesc(Im); colorbar;
figure; imagesc(-MultiScaleResult); colorbar;

%% problème de signe sur les GLRT...
%I1 =  Im-imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2); %newimage(I1)
%T1 = imfilter(I1.^2,ones(patch_half_size*2+1));
%%
%T1 = imfilter(Im.^2,ones(patch_half_size*2+1))+(patch_half_size*2+1)^2*imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2).^2-2*imfilter(Im,ones(patch_half_size*2+1)).*imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2);
T1 = imfilter(Im.^2,ones(patch_half_size*2+1))-(patch_half_size*2+1)^2*imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2).^2;

%%
P = (M'*M+mu*eye(profil_ext))^(-1)*(M');
Profiles = zeros(size(Im,1),size(Im,2),size(P,1));
for i=1:size(P,1)
    Profiles(:,:,i) = imfilter(Im,reshape(P(i,:),[patch_half_size*2+1 patch_half_size*2+1]));
end
E1 = imfilter(Im.^2,ones(patch_half_size*2+1));
Mf = full(M);
E2 = 0*E1;
for i=1:size(M,2)
    E2 = E2-2*squeeze(Profiles(:,:,i)).*imfilter(Im,reshape(Mf(:,i),[patch_half_size*2+1 patch_half_size*2+1]));
end
[U,S,V] = svd(full(M),'econ');
Proj = 0*Profiles;
for i=1:size(V,1)
    Proj(:,:,i) = S(i,i)*sum(Profiles.*reshape(V(:,i),[1,1,size(V,1)]),3);
end
E3 = sum(Proj.^2,3);
T2 = E1+E2+E3;

%%
figure;
imagesc([T1,T2]); colorbar;
figure;
imagesc([E1,E2,E3]); colorbar;
figure;
imagesc(T2-T1); colorbar;

%% regardons en un point:
i = 537;
j = 373;
v = Im(i-patch_half_size:i+patch_half_size,j-patch_half_size:j+patch_half_size);
e1 = sum((v(:)-mean(v(:))).^2)
p = P*v(:);
u = reshape(M*p,size(v));
newimage([v mean(v(:))*(0*v+1) u]);
e2 = sum((v(:)-u(:)).^2)

%%
I2 =  Im-; newimage(I2);


T1 = imfilter((Im-imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2)).^2,ones(patch_half_size*2+1));
