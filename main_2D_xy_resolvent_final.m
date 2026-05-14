clearvars;
close all;
clc

% params.kx=0; 
params.kz=1;
params.omega=0;
Ny=32;
Nx=24;
params.Ny=Ny;
params.Nx=Nx;
params.Re=358;
params.Ly=2;
params.Lx=2*pi;
params.gmres_tol=1e-6;
params.gmres_restart=[];
params.gmres_maxit=4*Nx*Ny;
params.gmres_verbose=false;
run_single_case=false;
run_benchmark=true;
compare_full_matrix=true;
benchmark_grid=[8 8;
                12 10;
                24 16;
                32 24];
% Preconditioner options:
%   'none'                 : no GMRES preconditioner
%   'fast_scalar_laplacian': LU of scalar Chebyshev Laplacian from chebdif
%   'pressure_correction_lu': velocity inverse plus pressure Schur correction
params.preconditioner_type='pressure_correction_lu';
params=prepare_params(params);

nsv = 1;

opts.Tolerance = 1e-3;
opts.MaxIterations = 30;
opts.SubspaceDimension = 8;

if run_single_case
    params_single = add_laplacian_preconditioner(params);
    H_mf = @(f,tflag) H_fun(f,tflag,params_single);

    [U_mf,S_mf,V_mf] = svds(H_mf,[3*Nx*Ny,3*Nx*Ny],nsv,'largest',opts);
    sigma_mf = diag(S_mf);
    disp(table(sigma_mf,'VariableNames',{'sigma_matrix_free'}))

    if compare_full_matrix
        [H,L]=resolvent_2D_xy(params);
        [U_full,S_full,V_full] = svds(H,nsv,'largest');
        sigma_full = diag(S_full);
        sigma_rel_err = abs(sigma_mf-sigma_full)./abs(sigma_full);

        [U_mf_aligned,u_err] = align_singular_vectors(U_mf,U_full);
        [V_mf_aligned,v_err] = align_singular_vectors(V_mf,V_full);

        comparison_table = table(sigma_mf,sigma_full,sigma_rel_err,u_err.',v_err.', ...
            'VariableNames',{'sigma_matrix_free','sigma_full','sigma_relative_error', ...
            'left_vector_error','right_vector_error'});
        disp(comparison_table)

        figure;
        tiledlayout(2,2)

        nexttile
        bar(categorical({'matrix-free','full matrix'}),[sigma_mf(1),sigma_full(1)])
        ylabel('\sigma')
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
        ylabel('|u_1|')
        title('Left singular vector')
        grid on

        nexttile
        plot(abs(V_full(:,1)),'k-',LineWidth=1.5)
        hold on
        plot(abs(V_mf_aligned(:,1)),'r--',LineWidth=1.2)
        legend('full matrix','matrix-free','Location','best')
        xlabel('forcing index')
        ylabel('|v_1|')
        title('Right singular vector')
        grid on
    end
end

if run_benchmark
    benchmark_table = benchmark_svd_methods(params,benchmark_grid,nsv,opts);
    disp(benchmark_table)

    figure;
    tiledlayout(1,2)

    nexttile
    plot(benchmark_table.N,benchmark_table.full_total_time_s,'ko-',LineWidth=1.5)
    hold on
    plot(benchmark_table.N,benchmark_table.matrix_free_total_time_s,'rs--',LineWidth=1.5)
    legend('full matrix','matrix-free','Location','best')
    xlabel('N = Nx Ny')
    ylabel('time (s)')
    title('SVD timing')
    grid on

    nexttile
    plot(benchmark_table.N,benchmark_table.full_memory_MB,'ko-',LineWidth=1.5)
    hold on
    plot(benchmark_table.N,benchmark_table.matrix_free_memory_MB,'rs--',LineWidth=1.5)
    legend('full matrix','matrix-free','Location','best')
    xlabel('N = Nx Ny')
    ylabel('memory estimate (MB)')
    title('Dominant array memory')
    grid on
end


function benchmark_table = benchmark_svd_methods(params_template,grid_list,nsv,opts)
    nGrid = size(grid_list,1);
    Ny_values = grid_list(:,1);
    Nx_values = grid_list(:,2);
    N_values = Ny_values.*Nx_values;

    full_build_time_s = zeros(nGrid,1);
    full_svd_time_s = zeros(nGrid,1);
    full_total_time_s = zeros(nGrid,1);
    matrix_free_setup_time_s = zeros(nGrid,1);
    matrix_free_svd_time_s = zeros(nGrid,1);
    matrix_free_total_time_s = zeros(nGrid,1);
    sigma_full = zeros(nGrid,1);
    sigma_matrix_free = zeros(nGrid,1);
    full_memory_MB = zeros(nGrid,1);
    matrix_free_memory_MB = zeros(nGrid,1);

    for iGrid = 1:nGrid
        params_i = params_template;
        params_i.Ny = Ny_values(iGrid);
        params_i.Nx = Nx_values(iGrid);
        params_i.gmres_maxit = 4*params_i.Nx*params_i.Ny;
        params_i.gmres_verbose = false;
        params_i = prepare_params(params_i);
        op_size = [3*params_i.Nx*params_i.Ny,3*params_i.Nx*params_i.Ny];

        drawnow;
        tic
        [H_i,L_i] = resolvent_2D_xy(params_i);
        full_build_time_s(iGrid) = toc;

        tic
        [U_full_i,S_full_i,V_full_i] = svds(H_i,nsv,'largest',opts);
        full_svd_time_s(iGrid) = toc;
        full_total_time_s(iGrid) = full_build_time_s(iGrid)+full_svd_time_s(iGrid);
        sigma_full(iGrid) = S_full_i(1,1);

        full_memory_MB(iGrid) = workspace_bytes( ...
            H_i,L_i,U_full_i,S_full_i,V_full_i)/1024^2;

        clear H_i U_full_i S_full_i V_full_i
        drawnow;

        tic
        params_i = add_laplacian_preconditioner(params_i);
        matrix_free_setup_time_s(iGrid) = toc;

        H_mf_i = @(f,tflag) H_fun(f,tflag,params_i);
        tic
        [U_mf_i,S_mf_i,V_mf_i] = svds(H_mf_i,op_size,nsv,'largest',opts);
        matrix_free_svd_time_s(iGrid) = toc;
        matrix_free_total_time_s(iGrid) = matrix_free_setup_time_s(iGrid) ...
            + matrix_free_svd_time_s(iGrid);
        sigma_matrix_free(iGrid) = S_mf_i(1,1);

        matrix_free_memory_MB(iGrid) = (workspace_bytes( ...
            params_i,U_mf_i,S_mf_i,V_mf_i)+matrix_free_workspace_bytes(params_i))/1024^2;

        clear H_mf_i U_mf_i S_mf_i V_mf_i L_i
    end

    sigma_relative_error = abs(sigma_matrix_free-sigma_full)./abs(sigma_full);

    benchmark_table = table(Ny_values,Nx_values,N_values, ...
        full_build_time_s,full_svd_time_s,full_total_time_s, ...
        matrix_free_setup_time_s,matrix_free_svd_time_s, ...
        matrix_free_total_time_s,full_memory_MB,matrix_free_memory_MB, ...
        sigma_full,sigma_matrix_free,sigma_relative_error, ...
        'VariableNames',{'Ny','Nx','N','full_build_time_s', ...
        'full_svd_time_s','full_total_time_s', ...
        'matrix_free_setup_time_s','matrix_free_svd_time_s', ...
        'matrix_free_total_time_s','full_memory_MB', ...
        'matrix_free_memory_MB','sigma_full','sigma_matrix_free', ...
        'sigma_relative_error'});
end


function bytes = matrix_free_workspace_bytes(params)
    n = 4*params.Nx*params.Ny;
    restart = params.gmres_restart;
    if isempty(restart)
        restart = min(params.gmres_maxit,n);
    end

    % Complex double Krylov basis used by GMRES, plus a few work vectors.
    bytes_per_complex_double = 16;
    bytes = bytes_per_complex_double*(n*(restart+1)+6*n);
    if strcmp(params.preconditioner_type,'fast_scalar_laplacian') ...
            || strcmp(params.preconditioner_type,'pressure_correction_lu')
        bytes = bytes+params.laplacian_preconditioner_nnz*(16+8);
    end
end


function params=prepare_params(params)
    if ~isfield(params,'preconditioner_type')
        params.preconditioner_type='none';
    end
    Ny=params.Ny;
    Nx=params.Nx;
    [params.cheb_y,~]=chebdif(Ny,1);
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
        {'none','fast_scalar_laplacian','pressure_correction_lu'});

    if strcmp(params.preconditioner_type,'none')
        return
    end

    Ny=params.Ny;
    Nx=params.Nx;
    N=Nx*Ny;

    [~,DM]=chebdif(Ny,2);
    Dyy=(2/params.Ly)^2*DM(:,:,2);

    [~,Dxx]=fourdif(Nx,2);
    Dxx=(2*pi/params.Lx)^2*Dxx;

    Dxx_2D=kron(Dxx,speye(Ny));
    Dyy_2D=kron(speye(Nx),Dyy);

    lap=Dxx_2D+Dyy_2D-(params.kz^2)*speye(N);

    bc_left_ind=1:Ny:N;
    bc_right_ind=Ny:Ny:N;
    lap(bc_left_ind,:)=0;
    lap(sub2ind(size(lap),bc_left_ind,bc_left_ind))=1;
    lap(bc_right_ind,:)=0;
    lap(sub2ind(size(lap),bc_right_ind,bc_right_ind))=1;
    params.laplacian_factor=decomposition(lap,'lu');
    params.laplacian_preconditioner_nnz=nnz(lap);

    if strcmp(params.preconditioner_type,'pressure_correction_lu')
        alpha_x=2*pi/params.Lx;
        alpha_y=2/params.Ly;
        params.pressure_precon_momentum_scale=1+abs(params.omega) ...
            +(params.kz^2+alpha_x^2+alpha_y^2)/params.Re;
    end
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
    restart = params.gmres_restart;
    maxit = params.gmres_maxit;


    if isfield(params,'preconditioner_type') && ~strcmp(params.preconditioner_type,'none')
        [L_inv_u_p,flag,relres,iter] = gmres(@(u_p) L_fun(u_p,tflag,params),Bf, ...
            restart,tol,maxit,@(rhs) laplacian_preconditioner_fun(rhs,params));
    else
        [L_inv_u_p,flag,relres,iter] = gmres(@(u_p) L_fun(u_p,tflag,params),Bf, ...
            restart,tol,maxit);
    end
    if isfield(params,'gmres_verbose') && params.gmres_verbose
        if isscalar(iter)
            iter_msg = sprintf('%g',iter);
        else
            iter_msg = sprintf('[%d %d]',iter(1),iter(2));
        end
        fprintf('GMRES %s: flag=%d, relres=%g, iter=%s\n', ...
            tflag,flag,relres,iter_msg);
    end
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
    if strcmp(params.preconditioner_type,'fast_scalar_laplacian')
        z(1:N)=params.laplacian_factor\rhs(1:N);
        z(N+1:2*N)=params.laplacian_factor\rhs(N+1:2*N);
        z(2*N+1:3*N)=params.laplacian_factor\rhs(2*N+1:3*N);
    elseif strcmp(params.preconditioner_type,'pressure_correction_lu')
        z=pressure_correction_preconditioner_fun(rhs,params);
        return
    else
        error('Unknown preconditioner type "%s".',params.preconditioner_type)
    end
    z(3*N+1:4*N)=rhs(3*N+1:4*N);
end


function z=pressure_correction_preconditioner_fun(rhs,params)
    Ny=params.Ny;
    Nx=params.Nx;
    N=Nx*Ny;
    kz=params.kz;
    alpha=params.pressure_precon_momentum_scale;

    ru=rhs(1:N);
    rv=rhs(N+1:2*N);
    rw=rhs(2*N+1:3*N);
    rp=rhs(3*N+1:4*N);

    u0=params.laplacian_factor\ru;
    v0=params.laplacian_factor\rv;
    w0=params.laplacian_factor\rw;

    div0=-Dx(u0,params)-Dy(v0,params)-1i*kz*w0;
    pressure_rhs=rp-div0;
    p=-alpha*(params.laplacian_factor\pressure_rhs);

    u=(u0-Dx(p,params)/alpha);
    v=(v0-Dy(p,params)/alpha);
    w=(w0-1i*kz*p/alpha);

    left_bc=1:Ny:N;
    right_bc=Ny:Ny:N;
    u(left_bc)=ru(left_bc);
    u(right_bc)=ru(right_bc);
    v(left_bc)=rv(left_bc);
    v(right_bc)=rv(right_bc);
    w(left_bc)=rw(left_bc);
    w(right_bc)=rw(right_bc);

    z=[u;v;w;p];
end

%first and second order derative in y, make sure to have scaling factor.
function df_dy=Dy(f,params)
    f_mat=reshape(f,params.Ny,params.Nx);
    df_dy_mat=cell2mat(cellfun(@(col) chebdifft(col,1), ...
        num2cell(f_mat,1),'UniformOutput',false));
    df_dy=reshape((2/params.Ly)*df_dy_mat,params.Ny*params.Nx,1);
end

function df_dyy=Dyy(f,params)
    f_mat=reshape(f,params.Ny,params.Nx);
    df_dyy_mat=cell2mat(cellfun(@(col) chebdifft(col,2), ...
        num2cell(f_mat,1),'UniformOutput',false));
    df_dyy=reshape((2/params.Ly)^2*df_dyy_mat,params.Ny*params.Nx,1);
    % 
    % df_dyy_mat = zeros(Ny,Nx);
    % 
    % for j = 1:Nx
    %     df_dyy_mat(:,j) = chebdifft(f_mat(:,j),2);
    % end

end

%first and second order derivative of x using fourdifft.
function df_dx=Dx(f,params)
    f_mat=reshape(f,params.Ny,params.Nx);
    df_dx_mat=cell2mat(cellfun(@(row) fourdifft(row.',1).', ...
        num2cell(f_mat,2),'UniformOutput',false));
    df_dx=reshape((2*pi/params.Lx)*df_dx_mat,params.Ny*params.Nx,1);
end

function df_dxx=Dxx(f,params)
    f_mat=reshape(f,params.Ny,params.Nx);
    df_dxx_mat=cell2mat(cellfun(@(row) fourdifft(row.',2).', ...
        num2cell(f_mat,2),'UniformOutput',false));
    df_dxx=reshape((2*pi/params.Lx)^2*df_dxx_mat,params.Ny*params.Nx,1);
end


function L_u_p=L_fun(u_p,tflag,params)
    % kx=params.kx;
    kz=params.kz;
    omega=params.omega;
    Ny=params.Ny;
    Nx=params.Nx;
    Re=params.Re;
    % Ly=params.Ly;
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
