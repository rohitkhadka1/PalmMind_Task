from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi.responses import JSONResponse

from ..db import get_db
from ..models import Booking
from ..schemas import BookingCreate, BookingOut


router = APIRouter()


@router.get("/")
async def booking_root():
    """
    Root endpoint for booking API.
    """
    return JSONResponse({
        "message": "Interview Booking API",
        "endpoints": {
            "/create": "Create a new booking",
            "/list": "List all bookings"
        }
    })


@router.get("/create", summary="Get create endpoint information", description="Returns information about how to create a booking")
async def create_booking_info():
    return JSONResponse({
        "message": "This is a POST endpoint. Please send a POST request with the following JSON body:",
        "example_payload": {
            "name": "John Doe",
            "email": "john@example.com",
            "date": "2025-11-01",
            "time": "14:30"
        }
    })

@router.post("/create", response_model=BookingOut)
async def create_booking(payload: BookingCreate, db: AsyncSession = Depends(get_db)):
    booking = Booking(name=payload.name, email=payload.email, date=payload.date, time=payload.time)
    db.add(booking)
    await db.commit()
    await db.refresh(booking)
    return booking


@router.get("/list", response_model=list[BookingOut])
async def list_bookings(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Booking).order_by(Booking.created_at.desc()))).scalars().all()
    return rows

