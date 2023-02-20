
% img.png est une image que j'ai extraite de l'un de tes rapports

%% GLRT sur différentes échelles:
I = im2double(imread('img.png'));
I = log(I+1/255);
I = I(10:end-10,20:end-10,1);
Im = I;
Im_orig = Im;

MultiScaleResult = 0*Im_orig;

figure;
for scale = 6:-1:1
    Im = imresize(Im_orig,1/scale); % on redimensionne l'image en fonction de l'échelle
    patch_half_size = 4;            % taille du patch: (2*patch_half_size+1) x (2*patch_half_size+1)
    profil_ext = ceil(sqrt(2)*(patch_half_size+1)); % taille du demi-profil
    Cmin = 0*Im+inf;                % carte de détection optimale (min sur les angles, en chaque point)
    mu = 1E-2;                      % régul pour éviter l'inversion d'une matrice singulière
    angular_step = 1;               % pas angulaire (angles theta)
    for theta = 2*pi*(0:angular_step:180-angular_step)/360

        [X,Y] = meshgrid(-patch_half_size:patch_half_size,-patch_half_size:patch_half_size);

        % 0/ on commence par construire la matrice (sparse) M:
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

        % 1/ on calcule le profil:
        P = (M'*M+mu*eye(profil_ext))^(-1)*(M');
        Profiles = zeros(size(Im,1),size(Im,2),size(P,1)); % Profiles est un tableau 3D, le long de la 3ème dim on stocke le demi-profil
        for i=1:size(P,1)
            Profiles(:,:,i) = imfilter(Im,reshape(P(i,:),[patch_half_size*2+1 patch_half_size*2+1]));
        end

        % 2/ on interdit les valeurs inférieures à celle au centre du profil
        Profiles = max(Profiles,Profiles(:,:,1));

        % 3/ on calcule l'énergie des résidus (calcul efficace basé sur le dev des normes L2 carré en 3 termes)
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

        % E4 correspond à |v-u1|^2
        E4 = -(imfilter(Im.^2,ones(patch_half_size*2+1))-(patch_half_size*2+1)^2*imfilter(Im,ones(patch_half_size*2+1)/(patch_half_size*2+1)^2).^2);

        Cmin = min(Cmin,(E1+E2+E3+E4)); % mise à jour de la carte de GLRT
        imagesc(Cmin); title(sprintf('theta=%d',round(360*theta/(2*pi)))); drawnow;
    end
    MultiScaleResult = MultiScaleResult + imresize(Cmin,size(MultiScaleResult)); % combinaison des échelles
    imagesc(MultiScaleResult); title(sprintf('scale=%d',scale)); drawnow;
end
figure; imagesc(Im); colorbar; title('image de départ');
figure; imagesc(-MultiScaleResult); colorbar; title('carte critère centerline');
