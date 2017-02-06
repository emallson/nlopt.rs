extern crate libc;
#[macro_use]
extern crate enum_primitive;

mod ffi;
mod algo;

use std::slice;

use libc::*;
use enum_primitive::FromPrimitive;

use algo::Algorithm;

pub type Objective<'a, T> = fn(&[f64], &mut Option<&mut T>) -> (f64, Option<&'a [f64]>);

struct ObjectiveWrapper<'a, T: 'a>(&'a Objective<'a, T>, Option<&'a mut T>);

extern "C" fn obj_wrapper<T>(n: c_uint, x: *const c_double, grad_out: Option<*mut c_double>, data: *mut c_void) -> f64 {
    let xs = unsafe { slice::from_raw_parts(x, n as usize) };
    let &mut ObjectiveWrapper(obj, ref mut t)= unsafe { &mut *(data as *mut ObjectiveWrapper<T>) };

    let (f, mgrad) = obj(&xs, t);

    if let Some(out) = grad_out {
        let grad = unsafe { slice::from_raw_parts_mut(out, n as usize) };
        if let Some(grad_data) = mgrad {
            for (i, &g) in grad_data.into_iter().enumerate() {
                grad[i] = g;
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
            None => Err(NLError::Unknown)
        }
    } else {
        Err(NLError::Unknown)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ObjectiveType {
    Minimize,
    Maximize
}

pub struct Problem {
    inst: ffi::ProbInst,
    vars: usize,
}

impl Problem {
    pub fn new(algo: Algorithm, n: usize) -> Problem {
        let inst = unsafe {
            ffi::nlopt_create(algo as c_uint, n as c_uint)
        };

        return Problem {
            inst: inst,
            vars: n,
        };
    }

    pub fn set_objective<T>(&mut self, obj_type: ObjectiveType, obj: &Objective<T>, initial: Option<&mut T>) -> Res {
        to_res(unsafe { 
            match obj_type {
                ObjectiveType::Minimize => ffi::nlopt_set_min_objective(&mut self.inst, obj_wrapper::<T>, &mut ObjectiveWrapper(obj, initial)as *mut _ as *mut c_void),
                ObjectiveType::Maximize => ffi::nlopt_set_max_objective(&mut self.inst, obj_wrapper::<T>, &mut ObjectiveWrapper(obj, initial)as *mut _ as *mut c_void)
            } 
        })
    }
}
