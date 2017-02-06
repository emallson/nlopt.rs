extern crate libc;
#[macro_use]
extern crate enum_primitive;

mod ffi;
mod algo;

use std::slice;

use libc::*;
use enum_primitive::FromPrimitive;

use algo::Algorithm;

pub trait Objective<T>
    : Fn(&[f64], &Option<&mut T>, bool) -> (f64, Option<Vec<f64>>) {
}

impl<T, O: Fn(&[f64], &Option<&mut T>, bool) -> (f64, Option<Vec<f64>>)> Objective<T> for O {}

struct ObjectiveWrapper<'a, T: 'a, O: 'a>(O, Option<&'a mut T>) where O: Objective<T>;

extern "C" fn obj_wrapper<T, O: Objective<T>>(n: c_uint,
                                              x: *const c_double,
                                              grad_out: *mut c_double,
                                              data: *mut c_void)
                                              -> f64 {
    let xs = unsafe { slice::from_raw_parts(x, n as usize) };
    let objdata = unsafe { &*(data as *const ObjectiveWrapper<T, O>) };
    let &ObjectiveWrapper(ref obj, ref t) = objdata;

    let grad_out = if grad_out == std::ptr::null_mut() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts_mut(grad_out, n as usize) })
    };


    let (f, mgrad) = obj(&xs, t, grad_out.is_some());

    if let Some(out) = grad_out {
        if let Some(grad_data) = mgrad {
            for (i, g) in grad_data.into_iter().enumerate() {
                out[i] = g;
            }
        }
    }

    f
}

enum_from_primitive! {
    #[derive(Debug, PartialEq, Copy, Clone)]
    pub enum NLResult {
        Success = 1,
        StopvalReached,
        FTolReached,
        XTolReached,
        MaxEvalReached,
        MaxTimeReached,
    }
}

enum_from_primitive! { 
#[derive(Debug, PartialEq, Copy, Clone)]
    pub enum NLError {
        ForcedStop = -5,
        RoundoffLimited,
        OutOfMemory,
        InvalidArgs,
        Failure,
        Unknown
    }
}
type Res = Result<NLResult, NLError>;

fn to_res(v: c_int) -> Res {
    if v > 0 {
        NLResult::from_i32(v).ok_or(NLError::Unknown)
    } else if v < 0 {
        match NLError::from_i32(v) {
            Some(e) => Err(e),
            None => Err(NLError::Unknown),
        }
    } else {
        Err(NLError::Unknown)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ObjectiveType {
    Minimize,
    Maximize,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum StopCond<'a> {
    FTolRel(f64),
    FTolAbs(f64),
    XTolRel(&'a [f64]),
    XTolAbs(&'a [f64]),
    XTolAbs1(f64),
    StopVal(f64),
    MaxEvals(u32),
    MaxTime(f64),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ConstraintType {
    Inequality,
    Equality,
}

pub struct Problem {
    inst: *mut ffi::ProbInst,
    vars: usize,
}

impl Problem {
    pub fn new(algo: Algorithm, n: usize) -> Problem {
        let inst = unsafe { ffi::nlopt_create(algo as c_int, n as c_uint) };

        return Problem {
            inst: inst,
            vars: n,
        };
    }

    pub fn set_objective<T, O: Objective<T>>(&mut self,
                                             obj_type: ObjectiveType,
                                             obj: O,
                                             initial: Option<&mut T>)
                                             -> Res {
        let obj_data = Box::new(ObjectiveWrapper(obj, initial));
        to_res(unsafe {
            match obj_type {
                ObjectiveType::Minimize => {
                    ffi::nlopt_set_min_objective(self.inst,
                                                 obj_wrapper::<T, O>,
                                                 Box::into_raw(obj_data) as *mut c_void)
                }
                ObjectiveType::Maximize => {
                    ffi::nlopt_set_max_objective(self.inst,
                                                 obj_wrapper::<T, O>,
                                                 Box::into_raw(obj_data) as *mut c_void)
                }
            }
        })
    }

    pub fn set_lower_bounds(&mut self, lb: &[f64]) -> Res {
        to_res(unsafe { ffi::nlopt_set_lower_bounds(self.inst, lb.as_ptr()) })
    }

    pub fn set_upper_bounds(&mut self, ub: &[f64]) -> Res {
        to_res(unsafe { ffi::nlopt_set_upper_bounds(self.inst, ub.as_ptr()) })
    }

    pub fn set_lower_bound(&mut self, lb: f64) -> Res {
        to_res(unsafe { ffi::nlopt_set_lower_bounds1(self.inst, lb) })
    }

    pub fn set_upper_bound(&mut self, ub: f64) -> Res {
        to_res(unsafe { ffi::nlopt_set_upper_bounds1(self.inst, ub) })
    }

    pub fn add_constraint<T, O: Objective<T>>(&mut self,
                                              ty: ConstraintType,
                                              con: O,
                                              initial: Option<&mut T>,
                                              tolerance: f64)
                                              -> Res {
        use ConstraintType::*;
        let objdata = Box::new(ObjectiveWrapper(con, initial));
        to_res(unsafe {
            match ty {
                Inequality => {
                    ffi::nlopt_add_inequality_constraint(self.inst,
                                                         obj_wrapper::<T, O>,
                                                         Box::into_raw(objdata) as *mut c_void,
                                                         tolerance)
                }
                Equality => {
                    ffi::nlopt_add_equality_constraint(self.inst,
                                                       obj_wrapper::<T, O>,
                                                       Box::into_raw(objdata) as *mut c_void,
                                                       tolerance)
                }
            }
        })
    }

    pub fn remove_constraints(&mut self, ty: ConstraintType) -> Res {
        use ConstraintType::*;
        to_res(unsafe {
            match ty {
                Inequality => ffi::nlopt_remove_inequality_constraints(self.inst),
                Equality => ffi::nlopt_remove_equality_constraints(self.inst),
            }
        })
    }

    pub fn set_stop(&mut self, cnd: StopCond) -> Res {
        use StopCond::*;
        to_res(unsafe {
            match cnd {
                StopVal(v) => ffi::nlopt_set_stopval(self.inst, v),
                MaxEvals(v) => ffi::nlopt_set_maxeval(self.inst, v as c_int),
                MaxTime(v) => ffi::nlopt_set_maxtime(self.inst, v),
                FTolRel(v) => ffi::nlopt_set_ftol_rel(self.inst, v),
                FTolAbs(v) => ffi::nlopt_set_ftol_abs(self.inst, v),
                XTolRel(v) => ffi::nlopt_set_xtol_rel(self.inst, v.as_ptr()),
                XTolAbs(v) => ffi::nlopt_set_xtol_abs(self.inst, v.as_ptr()),
                XTolAbs1(v) => ffi::nlopt_set_xtol_abs1(self.inst, v),
            }
        })
    }

    pub fn force_stop(&self) -> Res {
        to_res(unsafe { ffi::nlopt_force_stop(self.inst) })
    }

    pub fn optimize(&mut self, x: &[f64]) -> Result<(NLResult, Vec<f64>, f64), NLError> {
        let mut opt = 0.0;
        let mut x = Vec::from(x).clone();
        let res = to_res(unsafe { ffi::nlopt_optimize(self.inst, x.as_mut_ptr(), &mut opt) });
        res.map(|r| (r, x, opt))
    }
}

impl Drop for Problem {
    fn drop(&mut self) {
        unsafe {
            ffi::nlopt_destroy(self.inst);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use algo::Algorithm;

    #[test]
    fn tutorial() {
        let mut prob = Problem::new(Algorithm::MMA, 2);

        fn obj(x: &[f64], data: &Option<&mut ()>, gradp: bool) -> (f64, Option<Vec<f64>>) {
            let grad = match gradp {
                true => Some(vec![0.0, 0.5 / x[1].sqrt()]),
                false => None,
            };

            (x[1].sqrt(), grad)
        }

        struct Data {
            a: f64,
            b: f64,
        }

        fn constr(x: &[f64], data: &Option<&mut Data>, gradp: bool) -> (f64, Option<Vec<f64>>) {
            if let &Some(&mut Data { a, b }) = data {
                let grad = match gradp {
                    true => Some(vec![3.0 * a * (a * x[0] + b).powi(2), -1.0]),
                    false => None,
                };
                ((a * x[0] + b).powi(3) - x[1], grad)
            } else {
                println!("uhoh");
                panic!();
            }
        }

        prob.set_objective(ObjectiveType::Minimize, obj, None).unwrap();
        println!("test");
        prob.set_lower_bounds(&[std::f64::NEG_INFINITY, 0.0]).unwrap();
        prob.add_constraint(ConstraintType::Inequality,
                            constr,
                            Some(&mut Data { a: 2.0, b: 0.0 }),
                            1e-8)
            .unwrap();
        prob.add_constraint(ConstraintType::Inequality,
                            constr,
                            Some(&mut Data { a: -1.0, b: 1.0 }),
                            1e-8)
            .unwrap();

        prob.set_stop(StopCond::XTolRel(&[1e-4, 1e-4])).unwrap();

        let (res, x, opt) = prob.optimize(&[1.234, 5.678]).unwrap();

        println!("{:?} {:?} {}", res, x, opt);
        let true_x = vec![0.33334, 0.296296];
        let true_opt = 0.544330847;
        assert!((x[0] - true_x[0]).abs() <= 1e-3);
        assert!((x[1] - true_x[1]).abs() <= 1e-3);
        assert!((opt - true_opt).abs() <= 1e-3);
    }
}
