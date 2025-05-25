% numericalmethods.m

clear; clc;
pkg load symbolic

% --- Display Menu ---
disp('Numerical Methods for Root Finding');
disp('-----------------------------------');
disp('1. Graphical');
disp('2. Incremental Search');
disp('3. Bisection');
disp('4. Regula-Falsi');
disp('5. Secant');
disp('6. Newton-Raphson');
method = input('Select method (1-6): ');

if ~ismember(method, 1:6)
    error('Invalid method selected. Please choose a number between 1 and 6.');
end

func_str = input('Enter function f(x) (e.g., exp(-x)-x): ', 's');
f = @(x) eval(vectorize(func_str));

x_start = input('Enter start x: ');
x_end = input('Enter end x: ');

if method == 2
    dx = input('Enter step size (dx): ');
end
if method >= 3 && method <= 5
    tol = input('Enter tolerance (e.g., 1e-6): ');
end
if method == 6
    deriv_str = input('Enter derivative f''(x): ', 's');
    df = @(x) eval(vectorize(deriv_str));
    x0 = input('Enter initial guess x0: ');
    tol = input('Enter tolerance (e.g., 1e-6): ');
    % Auto-adjust plot range if too narrow
    if abs(x_end - x_start) < 0.1
        disp('Plot range too narrow. Expanding to [x0-2, x0+2] for better visualization.');
        x_start = x0 - 2;
        x_end = x0 + 2;
    end
end

% Plot the function
xv = linspace(x_start, x_end, 400);
yv = f(xv);
figure; plot(xv, yv, 'b-', 'LineWidth', 2); hold on;
plot(xv, zeros(size(xv)), 'k--');
xlabel('x'); ylabel('f(x)');
title(['f(x) = ', func_str]);
grid on;

% --- Method Implementations ---
root_found = false;
root_val = NaN;
switch method
    case 1
        disp('Graphical method: inspect the plot for roots.');
        disp('   x         f(x)');
        for xi = linspace(x_start, x_end, 10)
            fprintf('%10.6f  %12.6f\n', xi, f(xi));
        end
    case 2
        n = input('Enter n (number of decimal places for error criterion): ');
        es = 0.5 * 10^(2-n); % Stopping criterion in percent
        disp('Incremental Search Table:');
        fprintf('Iter   x_l     dx      x_u      f(x_l)    f(x_u)    f(x_l)*f(x_u)   |e_a| %%   Remark\n');
        x_l = x_start;
        dx0 = dx;
        iter = 1;
        found = false;
        ea = NaN;
        prev_xu = NaN;
        while x_l < x_end
            x_u = x_l + dx;
            if x_u > x_end
                x_u = x_end;
            end
            fxl = f(x_l);
            fxu = f(x_u);
            prod = fxl * fxu;
            if iter > 1
                ea = abs((x_u - prev_xu)/x_u) * 100;
            else
                ea = NaN;
            end
            if prod < 0
                remark = 'Revert back to x_l & consider smaller interval';
                fprintf('%2d  %7.3f  %5.3f  %7.3f  %9.5f  %9.5f  %12.5f  %8.5f  %s\n', iter, x_l, dx, x_u, fxl, fxu, prod, ea, remark);
                dx = dx / 10;
                found = true;
                if ~isnan(ea) && ea <= es
                    root_found = true;
                    root_val = (x_l + x_u) / 2;
                    break
                end
                % Do not advance x_l, just continue with smaller dx
            else
                remark = 'Go to next interval';
                fprintf('%2d  %7.3f  %5.3f  %7.3f  %9.5f  %9.5f  %12.5f  %8.5f  %s\n', iter, x_l, dx, x_u, fxl, fxu, prod, ea, remark);
                x_l = x_u;
                dx = dx0; % Reset dx to original step size
            end
            prev_xu = x_u;
            iter = iter + 1;
            if iter > 100
                disp('Max iterations reached.'); break;
            end
        end
        if ~found
            disp('No root found in the given interval.');
        end
        % Plot the function and root (if found) in a single figure
        xv = linspace(x_start, x_end, 400);
        yv = f(xv);
        figure(1); clf; % Always use figure 1 and clear it
        plot(xv, yv, 'b-', 'LineWidth', 2); hold on;
        plot(xv, zeros(size(xv)), 'k--');
        xlabel('x'); ylabel('f(x)');
        title(['f(x) = ', func_str]);
        grid on;
        if root_found && ~isnan(root_val)
            plot(root_val, 0, 'ro', 'MarkerSize', 8, 'DisplayName', 'Root');
            y_offset = max(yv) * 0.05;
            text(root_val, y_offset, ['Root: ', num2str(root_val, '%.5f')], ...
                'Color', 'red', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        end
        hold off;
        if root_found && ~isnan(root_val)
            disp(['Root approximation: ', num2str(root_val)]);
        end
    case 3
        if f(x_start) * f(x_end) > 0
            warning('f(x_start) and f(x_end) must have opposite signs for Bisection. Plotting function only.');
            return;
        end
        disp('Bisection Method Table:');
        fprintf('Iter   x_l      x_r      x_u      f(x_l)    f(x_r)    |e_a| %%   f(x_l)*f(x_r)  Remark\n');
        x_l = x_start; x_u = x_end;
        iter = 1; err = inf;
        x_r_old = NaN;
        while true
            x_r = (x_l + x_u)/2;
            fxl = f(x_l); fxr = f(x_r); fxu = f(x_u);
            prod = fxl * fxr;
            if iter > 1
                err = abs((x_r - x_r_old)/x_r) * 100; % percent
            else
                err = NaN;
            end
            if prod > 0
                remark = '2nd subinterval';
            elseif prod < 0
                remark = '1st subinterval';
            else
                remark = 'Root found';
            end
            fprintf('%2d  %7.5f  %7.5f  %7.5f  %9.5f  %9.5f  %8.5f  %12.5f  %s\n', ...
                iter, x_l, x_r, x_u, fxl, fxr, err, prod, remark);
            if prod < 0
                x_u = x_r;
            else
                x_l = x_r;
            end
            if ~isnan(x_r_old) && err < tol
                break;
            end
            if abs(fxr) < eps
                break;
            end
            x_r_old = x_r;
            iter = iter + 1;
            if iter > 100
                disp('Max iterations reached.'); break;
            end
        end
        disp(['Root approximation: ', num2str(x_r)]);
        root_found = true;
        root_val = x_r;
    case 4
        if f(x_start) * f(x_end) > 0
            warning('f(x_start) and f(x_end) must have opposite signs for Regula-Falsi. Plotting function only.');
            return;
        end
        disp('Regula-Falsi Method Table:');
        fprintf('Iter   x_l      x_u      x_r      f(x_l)    f(x_u)    f(x_r)    |e_a| %%   f(x_l)*f(x_r)  Remark\n');
        x_l = x_start; x_u = x_end;
        iter = 1; err = inf;
        x_r_old = NaN;
        while true
            fxl = f(x_l); fxu = f(x_u);
            x_r = (x_u*fxl - x_l*fxu)/(fxl - fxu);
            fxr = f(x_r);
            prod = fxl * fxr;
            if iter > 1
                err = abs((x_r - x_r_old)/x_r) * 100; % percent
            else
                err = NaN;
            end
            if prod > 0
                remark = 'xR = xL';
            elseif prod < 0
                remark = 'xR = xU';
            else
                remark = 'Root found';
            end
            fprintf('%2d  %7.5f  %7.5f  %7.5f  %9.5f  %9.5f  %9.5f  %8.5f  %12.5f  %s\n', ...
                iter, x_l, x_u, x_r, fxl, fxu, fxr, err, prod, remark);
            if abs(fxr) < eps
                break;
            end
            if prod < 0
                x_u = x_r;
            else
                x_l = x_r;
            end
            if ~isnan(x_r_old) && err < tol
                break;
            end
            x_r_old = x_r;
            iter = iter + 1;
            if iter > 100
                disp('Max iterations reached.'); break;
            end
        end
        disp(['Root approximation: ', num2str(x_r)]);
        root_found = true;
        root_val = x_r;
    case 5
        disp('Secant Method Table:');
        fprintf('Iter   x0       x1       x2       f(x0)     f(x1)     |e_a| %%\n');
        x0 = x_start; x1 = x_end;
        iter = 1; err = inf;
        while true
            fx0 = f(x0); fx1 = f(x1);
            if fx1 - fx0 == 0
                disp('Zero denominator, cannot proceed.'); break;
            end
            x2 = x1 - fx1*(x1 - x0)/(fx1 - fx0);
            if iter > 1
                err = abs((x2 - x1)/x2) * 100;
            else
                err = NaN;
            end
            fprintf('%2d  %7.5f  %7.5f  %7.5f  %9.5f  %9.5f  %8.5f\n', iter, x0, x1, x2, fx0, fx1, err);
            if ~isnan(err) && err < tol
                break;
            end
            x0 = x1;
            x1 = x2;
            iter = iter + 1;
            if iter > 100
                disp('Max iterations reached.'); break;
            end
        end
        disp(['Root approximation: ', num2str(x2)]);
        root_found = true;
        root_val = x2;
    case 6
        disp('Newton-Raphson Method Table:');
        fprintf('%-5s %10s %10s %10s %10s\n', 'Iter', 'x_i', 'f(x)', 'f''(x)', '|e_a|');
        xi = x0;
        iter = 1; err = inf;
        while true
            fxi = f(xi); dfxi = df(xi);
            if dfxi == 0
                disp('Zero derivative, cannot proceed.'); break;
            end
            xi_next = xi - fxi/dfxi;
            if iter > 1
                err = abs((xi_next - xi)/xi_next) * 100;
            else
                err = NaN;
            end
            if isnan(err)
                fprintf('%-5d %10.3f %10.3f %10.3f %10s\n', iter, xi, fxi, dfxi, '---');
            else
                fprintf('%-5d %10.3f %10.3f %10.3f %10.3f\n', iter, xi, fxi, dfxi, err);
            end
            if ~isnan(err) && err < tol
                break;
            end
            xi = xi_next;
            iter = iter + 1;
            if iter > 100
                disp('Max iterations reached.'); break;
            end
        end
        disp(['Root approximation: ', num2str(xi)]);
        root_found = true;
        root_val = xi;
end

% Plot root if found
if root_found && ~isnan(root_val)
    xv = linspace(x_start, x_end, 400);
    yv = f(xv);
    figure(1); clf; % Use and clear only one figure
    plot(xv, yv, 'b-', 'LineWidth', 2); hold on;
    plot(xv, zeros(size(xv)), 'k--');
    xlabel('x'); ylabel('f(x)');
    title(['f(x) = ', func_str]);
    grid on;
    if root_found && ~isnan(root_val)
        plot(root_val, 0, 'ro', 'MarkerSize', 8, 'DisplayName', 'Root');
        y_offset = max(yv) * 0.05;
        text(root_val, y_offset, ['Root: ', num2str(root_val, '%.5f')], ...
            'Color', 'red', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
    hold off;
end
