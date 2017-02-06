#![allow(dead_code)]
use libc::*;

pub enum ProbInst {}
pub type NLOptResult = c_int;

pub type NLFn = extern "C" fn(c_uint, *const c_double, *mut c_double, *mut c_void) -> c_double;
pub type NLMultiFn = extern "C" fn(c_uint,
                                   result: *mut c_double,
                                   c_uint,
                                   *const c_double,
                                   *mut c_double,
                                   *mut c_void);

#[allow(improper_ctypes)]
// it complains about using the above typedefs in signatures
#[link(name = "nlopt")]
#[link(name = "m")]
extern "C" {
    pub fn nlopt_create(algorithm: c_int, num_vars: c_uint) -> *mut ProbInst;
    pub fn nlopt_destroy(prob: *mut ProbInst);

    pub fn nlopt_set_min_objective(prob: *mut ProbInst,
                                   obj: NLFn,
                                   data: *mut c_void)
                                   -> NLOptResult;
    pub fn nlopt_set_max_objective(prob: *mut ProbInst,
                                   obj: NLFn,
                                   data: *mut c_void)
                                   -> NLOptResult;

    pub fn nlopt_set_lower_bounds(prob: *mut ProbInst, lb: *const c_double) -> NLOptResult;
    pub fn nlopt_set_upper_bounds(prob: *mut ProbInst, ub: *const c_double) -> NLOptResult;

    pub fn nlopt_set_lower_bounds1(prob: *mut ProbInst, lb: c_double) -> NLOptResult;
    pub fn nlopt_set_upper_bounds1(prob: *mut ProbInst, ub: c_double) -> NLOptResult;

    pub fn nlopt_add_inequality_constraint(prob: *mut ProbInst,
                                           fc: NLFn,
                                           data: *mut c_void,
                                           tol: c_double)
                                           -> NLOptResult;
    pub fn nlopt_add_equality_constraint(prob: *mut ProbInst,
                                         fc: NLFn,
                                         data: *mut c_void,
                                         tol: c_double)
                                         -> NLOptResult;

    pub fn nlopt_add_inequality_mconstraint(prob: *mut ProbInst,
                                            fc: NLMultiFn,
                                            data: *mut c_void,
                                            tol: *const c_double)
                                            -> NLOptResult;
    pub fn nlopt_add_equality_mconstraint(prob: *mut ProbInst,
                                          fc: NLMultiFn,
                                          data: *mut c_void,
                                          tol: *const c_double)
                                          -> NLOptResult;

    pub fn nlopt_remove_inequality_constraints(prob: *mut ProbInst) -> NLOptResult;
    pub fn nlopt_remove_equality_constraints(prob: *mut ProbInst) -> NLOptResult;

    pub fn nlopt_set_stopval(prob: *mut ProbInst, stopval: c_double) -> NLOptResult;
    pub fn nlopt_set_maxeval(prob: *mut ProbInst, maxeval: c_int) -> NLOptResult;
    pub fn nlopt_set_maxtime(prob: *mut ProbInst, maxtime: c_double) -> NLOptResult;

    pub fn nlopt_set_ftol_rel(prob: *mut ProbInst, tol: c_double) -> NLOptResult;
    pub fn nlopt_set_ftol_abs(prob: *mut ProbInst, tol: c_double) -> NLOptResult;
    pub fn nlopt_set_xtol_rel(prob: *mut ProbInst, tol: *const c_double) -> NLOptResult;
    pub fn nlopt_set_xtol_abs(prob: *mut ProbInst, tol: *const c_double) -> NLOptResult;
    pub fn nlopt_set_xtol_abs1(prob: *mut ProbInst, tol: c_double) -> NLOptResult;

    pub fn nlopt_force_stop(prob: *const ProbInst) -> NLOptResult;

    pub fn nlopt_optimize(prob: *mut ProbInst,
                          x: *mut c_double,
                          opt_f: *mut c_double)
                          -> NLOptResult;
}
