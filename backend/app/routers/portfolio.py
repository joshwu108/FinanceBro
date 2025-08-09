from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging

from app.services.database import get_db
from app.models.portfolio import Portfolio, PortfolioHolding
from app.services.data_collector import DataCollector
from app.services.price_broadcaster import price_broadcaster

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_portfolios(
    user_id: int = Query(1, description="User ID"),
    db: Session = Depends(get_db)
):
    """Get all portfolios for a user"""
    try:
        # Check if user exists, if not return empty list
        from app.models.user import User
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            logger.info(f"No user found with ID {user_id}, returning empty portfolio list")
            return {
                "portfolios": [],
                "total": 0
            }
        
        portfolios = db.query(Portfolio).filter(
            Portfolio.user_id == user_id,
            Portfolio.is_active == True
        ).all()
        
        return {
            "portfolios": [
                {
                    "id": portfolio.id,
                    "name": portfolio.name,
                    "description": portfolio.description,
                    "total_value": portfolio.total_value,
                    "total_return": portfolio.total_return,
                    "total_return_percent": portfolio.total_return_percent,
                    "created_at": portfolio.created_at.isoformat() if portfolio.created_at else None,
                    "updated_at": portfolio.updated_at.isoformat() if portfolio.updated_at else None
                }
                for portfolio in portfolios
            ],
            "total": len(portfolios)
        }
    except Exception as e:
        logger.error(f"Error fetching portfolios: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def create_portfolio(
    name: str,
    description: Optional[str] = None,
    user_id: int = Query(1, description="User ID"),
    db: Session = Depends(get_db)
):
    """Create a new portfolio"""
    try:
        # Check if user exists, if not create a default user
        from app.models.user import User
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            # Create a default user
            default_user = User(
                id=user_id,
                username=f"user{user_id}",
                email=f"user{user_id}@example.com",
                hashed_password="default_password_hash",  # In production, this should be properly hashed
                full_name=f"Default User {user_id}",
                is_active=True,
                is_verified=True
            )
            db.add(default_user)
            db.commit()
            db.refresh(default_user)
            logger.info(f"Created default user with ID {user_id}")
        
        portfolio = Portfolio(
            user_id=user_id,
            name=name,
            description=description
        )
        
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)
        
        return {
            "id": portfolio.id,
            "name": portfolio.name,
            "description": portfolio.description,
            "message": "Portfolio created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{portfolio_id}")
async def get_portfolio_details(
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed portfolio information including holdings"""
    try:
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        holdings = db.query(PortfolioHolding).filter(
            PortfolioHolding.portfolio_id == portfolio_id
        ).all()
        
        # Update current prices and calculate returns
        data_collector = DataCollector()
        updated_holdings = []
        total_current_value = 0
        total_cost = 0
        
        for holding in holdings:
            try:
                # Get current price
                stock_data = await data_collector.get_stock_data_yahoo(
                    holding.stock_symbol, period="1d", interval="1m"
                )
                
                if stock_data is not None and not stock_data.empty:
                    current_price = stock_data['close'].iloc[-1]
                    current_value = holding.shares * current_price
                    total_return = current_value - (holding.shares * holding.average_price)
                    total_return_percent = (total_return / (holding.shares * holding.average_price)) * 100
                    
                    # Update holding
                    holding.current_price = current_price
                    holding.current_value = current_value
                    holding.total_return = total_return
                    holding.total_return_percent = total_return_percent
                    holding.last_updated = datetime.utcnow()
                    
                    total_current_value += current_value
                    total_cost += holding.shares * holding.average_price
                else:
                    current_value = holding.current_value or 0
                    total_current_value += current_value
                    total_cost += holding.shares * holding.average_price
                
                updated_holdings.append({
                    "id": holding.id,
                    "stock_symbol": holding.stock_symbol,
                    "shares": holding.shares,
                    "average_price": holding.average_price,
                    "current_price": holding.current_price,
                    "current_value": holding.current_value,
                    "total_return": holding.total_return,
                    "total_return_percent": holding.total_return_percent,
                    "purchase_date": holding.purchase_date.isoformat() if holding.purchase_date else None
                })
                
            except Exception as e:
                logger.error(f"Error updating holding {holding.stock_symbol}: {str(e)}")
                continue
        
        # Update portfolio totals
        portfolio.total_value = total_current_value
        portfolio.total_return = total_current_value - total_cost
        portfolio.total_return_percent = (portfolio.total_return / total_cost * 100) if total_cost > 0 else 0
        portfolio.updated_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "portfolio": {
                "id": portfolio.id,
                "name": portfolio.name,
                "description": portfolio.description,
                "total_value": portfolio.total_value,
                "total_return": portfolio.total_return,
                "total_return_percent": portfolio.total_return_percent,
                "created_at": portfolio.created_at.isoformat() if portfolio.created_at else None,
                "updated_at": portfolio.updated_at.isoformat() if portfolio.updated_at else None
            },
            "holdings": updated_holdings,
            "total_holdings": len(updated_holdings)
        }
        
    except Exception as e:
        logger.error(f"Error fetching portfolio details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{portfolio_id}/holdings")
async def add_holding(
    portfolio_id: int,
    stock_symbol: str,
    shares: float,
    average_price: float,
    purchase_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Add a new holding to a portfolio"""
    try:
        # Verify portfolio exists
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Parse purchase_date if provided
        parsed_purchase_date = None
        if purchase_date:
            try:
                parsed_purchase_date = datetime.fromisoformat(purchase_date.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Invalid purchase_date format: {purchase_date}, using current date")
                parsed_purchase_date = datetime.utcnow()
        else:
            parsed_purchase_date = datetime.utcnow()
        
        # Create holding
        holding = PortfolioHolding(
            portfolio_id=portfolio_id,
            stock_symbol=stock_symbol.upper(),
            shares=shares,
            average_price=average_price,
            purchase_date=parsed_purchase_date
        )
        
        db.add(holding)
        db.commit()
        db.refresh(holding)
        
        # Trigger portfolio update via WebSocket
        await price_broadcaster.broadcast_portfolio_update(portfolio_id)
        
        return {
            "id": holding.id,
            "stock_symbol": holding.stock_symbol,
            "shares": holding.shares,
            "average_price": holding.average_price,
            "message": "Holding added successfully"
        }
    except Exception as e:
        logger.error(f"Error adding holding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{portfolio_id}/holdings/{holding_id}")
async def update_holding(
    portfolio_id: int,
    holding_id: int,
    shares: Optional[float] = None,
    average_price: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Update an existing holding"""
    try:
        holding = db.query(PortfolioHolding).filter(
            PortfolioHolding.id == holding_id,
            PortfolioHolding.portfolio_id == portfolio_id
        ).first()
        
        if not holding:
            raise HTTPException(status_code=404, detail="Holding not found")
        
        if shares is not None:
            holding.shares = shares
        if average_price is not None:
            holding.average_price = average_price
        
        holding.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(holding)
        
        return {
            "id": holding.id,
            "stock_symbol": holding.stock_symbol,
            "shares": holding.shares,
            "average_price": holding.average_price,
            "message": "Holding updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating holding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{portfolio_id}/holdings/{holding_id}")
async def delete_holding(
    portfolio_id: int,
    holding_id: int,
    db: Session = Depends(get_db)
):
    """Delete a holding from a portfolio"""
    try:
        holding = db.query(PortfolioHolding).filter(
            PortfolioHolding.id == holding_id,
            PortfolioHolding.portfolio_id == portfolio_id
        ).first()
        
        if not holding:
            raise HTTPException(status_code=404, detail="Holding not found")
        
        db.delete(holding)
        db.commit()
        
        return {"message": "Holding deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting holding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 