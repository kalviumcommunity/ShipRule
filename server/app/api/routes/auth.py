"""
ShipRule CDLP - Authentication API Routes
==========================================
Provides login and signup endpoints backed by MongoDB user storage.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status as http_status
from pydantic import BaseModel, Field, EmailStr

from app.core.config import settings
from app.api.dependencies import create_access_token, get_current_user
from app.db.mongodb import create_user, find_user_by_email, verify_password, update_user_password

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    """User login request payload."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class SignupRequest(BaseModel):
    """User registration request payload."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=4, description="User password")
    full_name: Optional[str] = Field(default="", description="User full name")


class ChangePasswordRequest(BaseModel):
    """Change password request payload."""
    old_password: str = Field(..., description="Current user password")
    new_password: str = Field(..., min_length=4, description="New user password")


class MasterLoginRequest(BaseModel):
    """Master Admin login request payload."""
    access_key: str = Field(..., description="Master Admin access key")



class AuthResponse(BaseModel):
    """Authentication token response payload."""
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


@router.post("/signup", response_model=AuthResponse, status_code=http_status.HTTP_201_CREATED)
def user_signup(payload: SignupRequest):
    """
    Registers a new user in MongoDB and returns a signed JWT bearer token.
    """
    email_clean = payload.email.strip().lower()
    if not email_clean or "@" not in email_clean:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid email address format."
        )

    if not payload.password or len(payload.password) < 4:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 4 characters long."
        )

    try:
        user_info = create_user(
            email=email_clean,
            password=payload.password,
            full_name=payload.full_name or "",
            role="user"
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )

    token = create_access_token({"sub": email_clean, "role": user_info["role"], "email": email_clean})
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        role=user_info["role"],
        email=email_clean
    )


@router.post("/login", response_model=AuthResponse)
def user_login(payload: LoginRequest):
    """
    Authenticates user credentials strictly against MongoDB user database and returns a signed JWT bearer token.
    """
    email_clean = payload.email.strip().lower()
    db_user = find_user_by_email(email_clean)

    if not db_user or not verify_password(payload.password, db_user.get("password_hash", "")):
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    token = create_access_token({"sub": email_clean, "role": db_user.get("role", "user"), "email": email_clean})
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        role=db_user.get("role", "user"),
        email=email_clean
    )



@router.post("/master-login", response_model=AuthResponse)
def master_admin_login(payload: MasterLoginRequest):
    """
    Authenticates Master Admin access key and returns an admin-privileged JWT token.
    """
    if payload.access_key.strip() != settings.ADMIN_ACCESS_KEY:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Master Admin access key"
        )

    token = create_access_token({"sub": "master_admin", "role": "admin", "email": "admin@shiprule.local"})
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        role="admin",
        email="admin@shiprule.local"
    )


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    """Returns details for current authenticated session."""
    return user


@router.post("/change-password")
def change_password(payload: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    """Updates user password in database."""
    user_email = user.get("email") or user.get("sub")
    if not user_email or user_email == "master_admin":
        user_email = user.get("email", "admin@shiprule.local")

    try:
        update_user_password(user_email, payload.old_password, payload.new_password)
        return {"status": "success", "message": "Password changed successfully."}
    except ValueError as val_err:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )

