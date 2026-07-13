clearvars;
close all;
clc

% params.kx=0; 
params.kz=1;
params.omega=0;
Ny=4;
Nx=4;
params.Ny=Ny;
params.Nx=Nx;
params.Re=358;
params.Ly=2;
params.Lx=2*pi;
params.gmres_tol=1e-6;
compare_full_matrix=true;
% Preconditioner options:
%   'fft_1d_laplacian'    : FFT in x and LU of 1D Chebyshev Laplacian per mode
params.preconditioner_type='fft_1d_laplacian';
params=prepare_params(params);

run_grid_benchmark = true;
benchmark_Nx_list = [4 8 12 24 36];
benchmark_Ny_list = [4 8 12 24 36];
matrix_free_total_cpu_start = cputime;
matrix_free_setup_cpu_start = cputime;
params = add_laplacian_preconditioner(params);
matrix_free_setup_cpu_time_s = cputime-matrix_free_setup_cpu_start;
nsv = 1;
H_mf = @(f,tflag) H_fun(f,tflag,params);

opts.Tolerance = 1e-3;
opts.MaxIterations = 30;
opts.SubspaceDimension = 8;

matrix_free_svd_cpu_start = cputime;
[U_mf,S_mf,V_mf] = svds(H_mf,[3*Nx*Ny,3*Nx*Ny],nsv,'largest',opts);
matrix_free_svd_cpu_time_s = cputime-matrix_free_svd_cpu_start;
matrix_free_total_cpu_time_s = cputime-matrix_free_total_cpu_start;
sigma_mf = diag(S_mf);
disp(table(sigma_mf,'VariableNames',{'sigma_matrix_free'}))

if compare_full_matrix
    full_matrix_total_cpu_start = cputime;
    full_matrix_build_cpu_start = cputime;
    [H,L]=resolvent_2D_xy(params); %#ok<UNRCH>
    full_matrix_build_cpu_time_s = cputime-full_matrix_build_cpu_start;

    full_matrix_svd_cpu_start = cputime;
    [U_full,S_full,V_full] = svds(H,nsv,'largest');
    full_matrix_svd_cpu_time_s = cputime-full_matrix_svd_cpu_start;
    full_matrix_total_cpu_time_s = cputime-full_matrix_total_cpu_start;
    sigma_full = diag(S_full);
    sigma_rel_err = abs(sigma_mf-sigma_full)./abs(sigma_full);

    [U_mf_aligned,u_err] = align_singular_vectors(U_mf,U_full);
    [V_mf_aligned,v_err] = align_singular_vectors(V_mf,V_full);

    comparison_table = table(sigma_mf,sigma_full,sigma_rel_err,u_err.',v_err.', ...
        'VariableNames',{'sigma_matrix_free','sigma_full','sigma_relative_error', ...
        'left_vector_error','right_vector_error'});
    disp(comparison_table)

    cpu_time_table = table(matrix_free_setup_cpu_time_s, ...
        matrix_free_svd_cpu_time_s,matrix_free_total_cpu_time_s, ...
        full_matrix_build_cpu_time_s,full_matrix_svd_cpu_time_s, ...
        full_matrix_total_cpu_time_s, ...
        'VariableNames',{'matrix_free_preconditioner_setup_cpu_s', ...
        'matrix_free_svds_cpu_s','matrix_free_total_cpu_s', ...
        'full_matrix_build_cpu_s','full_matrix_svds_cpu_s', ...
        'full_matrix_total_cpu_s'});
    disp(cpu_time_table)

    figure;
    tiledlayout(2,2)

    nexttile
    bar(categorical({'matrix-free','full matrix'}),[sigma_mf(1),sigma_full(1)])
    ylabel('$\sigma$','Interpreter','latex')
    title('Largest singular value')

    nexttile
    semilogy(1:nsv,sigma_rel_err,'o-')
    xlabel('singular value index')
    ylabel('relative error')
    title('Singular value error')
    grid on

    nexttile
    plot(abs(U_full(:,1)),'k-',LineWidth=1.5)
    hold on
    plot(abs(U_mf_aligned(:,1)),'r--',LineWidth=1.2)
    legend('full matrix','matrix-free','Location','best')
    xlabel('state index')
    ylabel('$|u_1|$','Interpreter','latex')
    title('Left singular vector')
    grid on

    nexttile
    plot(abs(V_full(:,1)),'k-',LineWidth=1.5)
    hold on
    plot(abs(V_mf_aligned(:,1)),'r--',LineWidth=1.2)
    legend('full matrix','matrix-free','Location','best')
    xlabel('forcing index')
    ylabel('$|v_1|$','Interpreter','latex')
    title('Right singular vector')
    grid on
end

if run_grid_benchmark
    benchmark_table = benchmark_svd_methods(params,benchmark_Nx_list,benchmark_Ny_list,nsv,opts);
    disp(benchmark_table)

    figure;
    tiledlayout(1,2)

    nexttile
    loglog(benchmark_table.grid_points,benchmark_table.full_total_cpu_s,'ko-',LineWidth=1.5)
    hold on
    loglog(benchmark_table.grid_points,benchmark_table.matrix_free_total_cpu_s,'rs--',LineWidth=1.5)
    legend('full matrix','matrix-free','Location','best')
    xlabel('Nx Ny')
    ylabel('CPU time (s)')
    title('SVD CPU timing')
    grid on

    nexttile
    loglog(benchmark_table.grid_points,benchmark_table.full_memory_MB,'ko-',LineWidth=1.5)
    hold on
    loglog(benchmark_table.grid_points,benchmark_table.matrix_free_memory_MB,'rs--',LineWidth=1.5)
    legend('full matrix','matrix-free','Location','best')
    xlabel('Nx Ny')
    ylabel('memory estimate (MB)')
    title('Dominant array memory')
    grid on
end


function benchmark_table = benchmark_svd_methods(params_template,Nx_list,Ny_list,nsv,svds_opts)
    if numel(Nx_list) ~= numel(Ny_list)
        error('benchmark_svd_methods:gridListSizeMismatch', ...
            'Nx_list and Ny_list must have the same number of entries.');
    end

    Nx_cases = Nx_list(:);
    Ny_cases = Ny_list(:);
    nCases = numel(Nx_cases);

    grid_points = Nx_cases.*Ny_cases;
    state_size = 3*grid_points;
    linear_system_size = 4*grid_points;
    full_build_cpu_s = zeros(nCases,1);
    full_svds_cpu_s = zeros(nCases,1);
    full_total_cpu_s = zeros(nCases,1);
    matrix_free_setup_cpu_s = zeros(nCases,1);
    matrix_free_svds_cpu_s = zeros(nCases,1);
    matrix_free_total_cpu_s = zeros(nCases,1);
    sigma_full = zeros(nCases,1);
    sigma_matrix_free = zeros(nCases,1);
    full_memory_MB = zeros(nCases,1);
    matrix_free_memory_MB = zeros(nCases,1);

    for iCase = 1:nCases
        params_i = params_template;
        params_i.Nx = Nx_cases(iCase);
        params_i.Ny = Ny_cases(iCase);
        params_i = prepare_params(params_i);
        op_size = [3*params_i.Nx*params_i.Ny,3*params_i.Nx*params_i.Ny];

        drawnow;
        full_total_cpu_start = cputime;
        full_build_cpu_start = cputime;
        [H_i,L_i] = resolvent_2D_xy(params_i);
        full_build_cpu_s(iCase) = cputime-full_build_cpu_start;

        full_svds_cpu_start = cputime;
        [U_full_i,S_full_i,V_full_i] = svds(H_i,nsv,'largest',svds_opts);
        full_svds_cpu_s(iCase) = cputime-full_svds_cpu_start;
        full_total_cpu_s(iCase) = cputime-full_total_cpu_start;
        sigma_full(iCase) = S_full_i(1,1);

        full_memory_MB(iCase) = workspace_bytes( ...
            H_i,L_i,U_full_i,S_full_i,V_full_i)/1024^2;

        matrix_free_total_cpu_start = cputime;
        matrix_free_setup_cpu_start = cputime;
        params_i = add_laplacian_preconditioner(params_i);
        matrix_free_setup_cpu_s(iCase) = cputime-matrix_free_setup_cpu_start;

        clear H_i U_full_i S_full_i V_full_i
        drawnow;

        H_mf_i = @(f,tflag) H_fun(f,tflag,params_i);
        matrix_free_svds_cpu_start = cputime;
        [U_mf_i,S_mf_i,V_mf_i] = svds(H_mf_i,op_size,nsv,'largest',svds_opts);
        matrix_free_svds_cpu_s(iCase) = cputime-matrix_free_svds_cpu_start;
        matrix_free_total_cpu_s(iCase) = cputime-matrix_free_total_cpu_start;
        sigma_matrix_free(iCase) = S_mf_i(1,1);

        matrix_free_memory_MB(iCase) = (workspace_bytes( ...
            params_i,U_mf_i,S_mf_i,V_mf_i)+matrix_free_workspace_bytes(params_i))/1024^2;

        clear H_mf_i U_mf_i S_mf_i V_mf_i L_i
    end

    sigma_relative_error = abs(sigma_matrix_free-sigma_full)./abs(sigma_full);

    benchmark_table = table(Nx_cases,Ny_cases,grid_points,state_size, ...
        linear_system_size,full_build_cpu_s,full_svds_cpu_s,full_total_cpu_s, ...
        matrix_free_setup_cpu_s,matrix_free_svds_cpu_s,matrix_free_total_cpu_s, ...
        full_memory_MB,matrix_free_memory_MB,sigma_full,sigma_matrix_free, ...
        sigma_relative_error, ...
        'VariableNames',{'Nx','Ny','grid_points','state_size', ...
        'linear_system_size','full_build_cpu_s','full_svds_cpu_s', ...
        'full_total_cpu_s','matrix_free_setup_cpu_s', ...
        'matrix_free_svds_cpu_s','matrix_free_total_cpu_s', ...
        'full_memory_MB','matrix_free_memory_MB','sigma_full', ...
        'sigma_matrix_free','sigma_relative_error'});
end


function bytes = matrix_free_workspace_bytes(params)
    n = 4*params.Nx*params.Ny;
    restart = min(100,n);

    % Complex double Krylov basis used by GMRES, plus a few work vectors.
    bytes_per_complex_double = 16;
    bytes = bytes_per_complex_double*(n*(restart+1)+6*n);
    if strcmp(params.preconditioner_type,'fft_1d_laplacian')
        bytes = bytes+params.Nx*params.Ny^2*(16+8);
    end
end


function params=prepare_params(params)
    Ny=params.Ny;
    Nx=params.Nx;
    [params.cheb_y,DM]=chebdif(Ny,2);
    params.Dy_mat=(2/params.Ly)*DM(:,:,1);
    params.Dyy_mat=(2/params.Ly)^2*DM(:,:,2);

    [~,Dx_mat]=fourdif(Nx,1);
    [~,Dxx_mat]=fourdif(Nx,2);
    params.Dx_mat=(2*pi/params.Lx)*Dx_mat;
    params.Dxx_mat=(2*pi/params.Lx)^2*Dxx_mat;
    N=params.Ny*params.Nx;

    %vectorized velocity and velocity gradient that will be used to form
    %resolvent operators

    %for wavy wall, these mean flow and K_inv should be changed. Right now,
    %this is only the laminar Poiseuille flow. 
    params.U=reshape((1-params.cheb_y.^2)*ones(1,Nx),N,1);
    params.dUdy=reshape(-2*params.cheb_y*ones(1,Nx),N,1);
    params.dUdx=reshape(zeros(Ny,Nx),N,1);
    
    params.V=reshape(zeros(Ny,Nx),N,1);
    params.dVdx=reshape(zeros(Ny,Nx),N,1);
    params.dVdy=reshape(zeros(Ny,Nx),N,1);

    params.K_inv=reshape(zeros(Ny,Nx),N,1);

    [~,w]=clencurt(Ny-1);
    params.w=w(:);
    params.w_2D=reshape(params.w*ones(1,Nx),N,1);
    params.w_all=[params.w_2D;params.w_2D;params.w_2D];

end

function params = add_laplacian_preconditioner(params)
    params.preconditioner_type = validatestring(params.preconditioner_type, ...
        {'fft_1d_laplacian'});

    Ny=params.Ny;
    Nx=params.Nx;
    [~,DM]=chebdif(Ny,2);
    Dyy=(2/params.Ly)^2*DM(:,:,2);

    params.laplacian_1d_factors=cell(Nx,1);
    wave=fourier_wavenumbers(Nx);
    for kx_ind=1:Nx
        dxx_eigenvalue=-(2*pi/params.Lx)^2*wave(kx_ind)^2;
        lap_k=Dyy+(dxx_eigenvalue-params.kz^2)*speye(Ny);
        lap_k(1,:)=0;
        lap_k(1,1)=1;
        lap_k(Ny,:)=0;
        lap_k(Ny,Ny)=1;
        params.laplacian_1d_factors{kx_ind}=decomposition(lap_k,'lu');
    end
end


function wave=fourier_wavenumbers(Nx)
    N1=floor((Nx-1)/2);
    N2=(-Nx/2)*ones(rem(Nx+1,2));
    wave=[0:N1 N2 -N1:-1].';
end


function bytes = workspace_bytes(varargin)
    bytes = 0;
    for k = 1:nargin
        value = varargin{k}; %#ok<NASGU>
        info = whos('value');
        bytes = bytes+info.bytes;
    end
end


function [U_aligned,err] = align_singular_vectors(U,U_ref)
    U_aligned = U;
    err = zeros(1,size(U,2));
    for j = 1:size(U,2)
        phase = U_ref(:,j)'*U(:,j);
        if phase ~= 0
            U_aligned(:,j) = U(:,j)*conj(phase)/abs(phase);
        end
        err(j) = norm(U_aligned(:,j)-U_ref(:,j))/norm(U_ref(:,j));
    end
end


function H_f=H_fun(f,tflag,params)

    %kx,kz,omega,Ny,Re,Ly
    Ny=params.Ny;
    Nx=params.Nx;
    N=Nx*Ny;

    w_all=params.w_all;

    %if strcmp(tflag,'notransp')
        Bf=[f.*w_all.^(-1/2);
            zeros(N,1)];
    % else
    %     Bf=[f.*w_all.^(1/2);
    %         zeros(N,1)];
    % end
    
    %B.C. for u
    left_bc=1:Ny:N;
    right_bc=Ny:Ny:N;

    Bf(left_bc)=0;
    Bf(right_bc)=0;
    
    %B.C. for v
    Bf(N+left_bc)=0;
    Bf(N+right_bc)=0;

    %B.C. for w
    Bf(2*N+left_bc)=0;
    Bf(2*N+right_bc)=0;

    tol = params.gmres_tol;
    restart = min(100,4*Ny);
    maxit = 4*N;

    [L_inv_u_p,flag,relres,iter] = gmres(@(u_p) L_fun(u_p,tflag,params),Bf, ...
        restart,tol,maxit,@(rhs) laplacian_preconditioner_fun(rhs,params));
    if flag ~= 0
        if isscalar(iter)
            iter_msg = sprintf('%g',iter);
        else
            iter_msg = sprintf('[%d %d]',iter(1),iter(2));
        end
        warning('main_1D_resolvent:linearSolveNotConverged', ...
            'GMRES did not converge for %s solve. flag=%d, relres=%g, iter=%s.', ...
            tflag,flag,relres,iter_msg);
    end

   % if strcmp(tflag,'notransp')
        H_f = w_all.^(1/2).*L_inv_u_p(1:3*N,1);
    % else
    %     H_f = w_all.^(-1/2).*L_inv_u_p(1:3*N,1);
    % end
end


function z=laplacian_preconditioner_fun(rhs,params)
    Ny=params.Ny;
    Nx=params.Nx;
    N=Nx*Ny;

    z=zeros(4*N,1);
    z(1:N)=laplacian_1d_fft_solve(rhs(1:N),params);
    z(N+1:2*N)=laplacian_1d_fft_solve(rhs(N+1:2*N),params);
    z(2*N+1:3*N)=laplacian_1d_fft_solve(rhs(2*N+1:3*N),params);
    z(3*N+1:4*N)=rhs(3*N+1:4*N);
end


function z=laplacian_1d_fft_solve(rhs,params)
    Ny=params.Ny;
    Nx=params.Nx;

    rhs_hat=fft(reshape(rhs,Ny,Nx),[],2);
    z_hat=zeros(Ny,Nx);
    for kx_ind=1:Nx
        z_hat(:,kx_ind)=params.laplacian_1d_factors{kx_ind}\rhs_hat(:,kx_ind);
    end
    z=reshape(ifft(z_hat,[],2),Ny*Nx,1);
end

%first and second order derative in y, make sure to have scaling factor.
function df_dy=Dy(f,params)
    f_mat=reshape(f,params.Ny,params.Nx);
    df_dy=reshape(params.Dy_mat*f_mat,params.Ny*params.Nx,1);
end

function df_dyy=Dyy(f,params)
    f_mat=reshape(f,params.Ny,params.Nx);
    df_dyy=reshape(params.Dyy_mat*f_mat,params.Ny*params.Nx,1);
end

%first and second order derivative of x using fourdifft.
function df_dx=Dx(f,params)
    f_mat=reshape(f,params.Ny,params.Nx);
    df_dx=reshape(f_mat*params.Dx_mat.',params.Ny*params.Nx,1);
end

function df_dxx=Dxx(f,params)
    f_mat=reshape(f,params.Ny,params.Nx);
    df_dxx=reshape(f_mat*params.Dxx_mat.',params.Ny*params.Nx,1);
end

function L_u_p=L_fun(u_p,tflag,params)
    kz=params.kz;
    omega=params.omega;
    Ny=params.Ny;
    Nx=params.Nx;
    Re=params.Re;
    N=Nx*Ny;

    u=u_p(1:N);
    v=u_p(N+1:2*N);
    w=u_p(1+2*N:3*N);
    p=u_p(1+3*N:4*N);

    ux=Dx(u,params);
    vx=Dx(v,params);
    wx=Dx(w,params);
    px=Dx(p,params);

    uy=Dy(u,params);
    vy=Dy(v,params);
    wy=Dy(w,params);
    py=Dy(p,params);

    lap_u=Dyy(u,params)+Dxx(u,params)-(kz^2).*u;
    lap_v=Dyy(v,params)+Dxx(v,params)-(kz^2).*v;
    lap_w=Dyy(w,params)+Dxx(w,params)-(kz^2).*w;

    U = params.U;
    dUdx = params.dUdx;
    dUdy = params.dUdy;

    V = params.V;
    dVdx = params.dVdx;
    dVdy = params.dVdy;

    K_inv=params.K_inv;
    if strcmp(tflag,'notransp')
        L_u=1i*omega*u+U.*ux+V.*uy-lap_u/Re+K_inv.*u;
        L_v=1i*omega*v+U.*vx+V.*vy-lap_v/Re+K_inv.*v;
        L_w=1i*omega*w+U.*wx+V.*wy-lap_w/Re+K_inv.*w;
        L_u_p=[L_u+dUdx.*u+dUdy.*v+px;
               L_v+dVdx.*u+dVdy.*v+py;
               L_w+1i*kz*p;
            -ux-vy-1i*kz*w];

    else
        L_u=-1i*omega*u-U.*ux-V.*uy-lap_u/Re+K_inv.*u;
        L_v=-1i*omega*v-U.*vx-V.*vy-lap_v/Re+K_inv.*v;
        L_w=-1i*omega*w-U.*wx-V.*wy-lap_w/Re+K_inv.*w;
        L_u_p=[L_u+dUdx.*u+dVdx.*v+px;
               L_v+dUdy.*u+dVdy.*v+py;
               L_w+1i*kz*p;
            -ux-vy-1i*kz*w];

    end
    left_bc=1:Ny:N;
    right_bc=Ny:Ny:N;
    
    %B.C. for u
    L_u_p(left_bc)=u(left_bc);
    L_u_p(right_bc)=u(right_bc);
    
    %B.C. for v
    L_u_p(N+left_bc)=v(left_bc);
    L_u_p(N+right_bc)=v(right_bc);

    %B.C. for w
    L_u_p(2*N+left_bc)=w(left_bc);
    L_u_p(2*N+right_bc)=w(right_bc);

end


function [H,L]=resolvent_2D_xy(params)
    % kx=params.kx;
    kz=params.kz;
    omega=params.omega;
    Ny=params.Ny;
    Nx=params.Nx;
    Ly=params.Ly;
    Lx=params.Lx;

    N=Nx*Ny;
    Re=params.Re;

    Ix=speye(Nx);
    Iy=speye(Ny);

    [~,DM] = chebdif(Ny,2);
    D1 = (2/Ly)*DM(1:Ny,1:Ny,1);
    D2 = (2/Ly)^2*DM(1:Ny,1:Ny,2);

    %Dy_2D = kron(Ix,Dy);
    %Dyy_2D = kron(Ix,Dyy);

    [~,Dx] = fourdif(Nx,1);
    Dx = (2*pi/Lx)*Dx;

    [~,Dxx] = fourdif(Nx,2);
    Dxx = (2*pi/Lx)^2*Dxx; 
    %Dx_2D = kron(Dx,Iy);
    %Dyy_2D = kron(Dxx,Iy);

    I = speye(N);

    lplc = -kz^2*I + kron(Ix,D2) + kron(Dxx,Iy); % sparse

    % U0 = 1-y.^2;
    % U1 = -2*y;

    U = spdiags(params.U,0,N,N);
    dUdx = spdiags(params.dUdx,0,N,N);
    dUdy = spdiags(params.dUdy,0,N,N);

    V = spdiags(params.V,0,N,N);
    dVdx = spdiags(params.dVdx,0,N,N);
    dVdy = spdiags(params.dVdy,0,N,N);

    L11 = 1i*omega*I+U*kron(Dx,Iy) + V*kron(Ix,D1) + dUdx - (1/Re).*lplc;
    L12 = dUdy;
    L21 = dVdx;
    L14 = kron(Dx,Iy);
    L22 = 1i*omega*I+U*kron(Dx,Iy) + V*kron(Ix,D1) + dVdy - (1/Re)*lplc;
    L24 = kron(Ix,D1);
    L33 = 1i*omega*I+U*kron(Dx,Iy) + V*kron(Ix,D1) - (1/Re)*lplc;
    L34 = 1i*kz*I;
    L41 = -kron(Dx,Iy);
    L42 = -kron(Ix,D1);
    L43 = -1i*kz*kron(Ix,Iy);

    % Assemble A (sparse)
    L = [L11 L12 sparse(N,N) L14; ...
        L21 L22 sparse(N,N) L24; ...
        sparse(N,N) sparse(N,N) L33 L34; ...
        L41 L42 L43 sparse(N,N)];

    % B = spalloc(4*N,3*N, 3*N);
    % C = spalloc(3*N,4*N, 3*N);

    [~,w] = clencurt(Ny-1);

    w_sqrt = sqrt(w);
    w_sqrt = spdiags(w_sqrt',0,Ny,Ny);
    w_sqrt = kron(Ix, w_sqrt);
    
    w_sqrt_inv = 1./(sqrt(w)); % elementwise sqrt
    w_sqrt_inv = spdiags(w_sqrt_inv',0,Ny,Ny);
    w_sqrt_inv = kron(Ix, w_sqrt_inv);

    zero_matrix = spalloc(N,N, N);
    B = [w_sqrt_inv zero_matrix zero_matrix;
        zero_matrix w_sqrt_inv zero_matrix;
        zero_matrix zero_matrix w_sqrt_inv;
        zero_matrix zero_matrix zero_matrix];

    C = [w_sqrt zero_matrix zero_matrix zero_matrix;
        zero_matrix w_sqrt zero_matrix zero_matrix;
        zero_matrix zero_matrix w_sqrt zero_matrix];

    left_bc=1:Ny:N;
    right_bc=Ny:Ny:N;

    %B.C. of u
    L(left_bc,:)=0;
    L(sub2ind(size(L),left_bc,left_bc))=1;
    
    L(right_bc,:)=0;
    L(sub2ind(size(L),right_bc,right_bc))=1;

    B(left_bc,:)=0;
    B(right_bc,:)=0;

    %B.C. of v
    L(N+left_bc,:)=0;
    L(sub2ind(size(L),N+left_bc,N+left_bc))=1;
    
    L(N+right_bc,:)=0;
    L(sub2ind(size(L),N+right_bc,N+right_bc))=1;

    B(N+left_bc,:)=0;
    B(N+right_bc,:)=0;

      %B.C. of w
    L(2*N+left_bc,:)=0;
    L(sub2ind(size(L),2*N+left_bc,2*N+left_bc))=1;
    
    L(2*N+right_bc,:)=0;
    L(sub2ind(size(L),2*N+right_bc,2*N+right_bc))=1;

    B(2*N+left_bc,:)=0;
    B(2*N+right_bc,:)=0;

    H = C*(L\B);

end
