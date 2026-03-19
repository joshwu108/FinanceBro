from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging

from app.services.database import get_db
from app.models.alert import Alert, AlertType, AlertStatus
from app.services.data_collector import DataCollector

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_alerts(
    user_id: int = Query(1, description="User ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """Get all alerts for a user"""
    try:
        query = db.query(Alert).filter(Alert.user_id == user_id)
        
        if status:
            query = query.filter(Alert.alert_status == AlertStatus(status))
        
        alerts = query.all()
        
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "stock_symbol": alert.stock_symbol,
                    "alert_type": alert.alert_type.value,
                    "status": alert.alert_status.value,
                    "target_price": alert.target_price,
                    "message": alert.message,
                    "created_at": alert.created_at.isoformat() if alert.created_at else None,
                    "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else None
                }
                for alert in alerts
            ],
            "total": len(alerts)
        }
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def create_alert(
    stock_symbol: str,
    alert_type: str,
    target_price: Optional[float] = None,
    target_percent: Optional[float] = None,
    message: Optional[str] = None,
    user_id: int = Query(1, description="User ID"),
    db: Session = Depends(get_db)
):
    """Create a new alert"""
    try:
        # Validate alert type
        if alert_type not in [t.value for t in AlertType]:
            raise HTTPException(status_code=400, detail="Invalid alert type")
        
        # Create alert
        alert = Alert(
            user_id=user_id,
            stock_symbol=stock_symbol.upper(),
            alert_type=AlertType(alert_type),
            target_price=target_price,
            target_percent=target_percent,
            message=message or f"Alert for {stock_symbol.upper()}"
        )
        
        db.add(alert)
        db.commit()
        db.refresh(alert)
        
        return {
            "id": alert.id,
            "stock_symbol": alert.stock_symbol,
            "alert_type": alert.alert_type.value,
            "status": alert.alert_status.value,
            "message": "Alert created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{alert_id}")
async def update_alert(
    alert_id: int,
    alert_type: Optional[str] = None,
    target_price: Optional[float] = None,
    target_percent: Optional[float] = None,
    message: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Update an existing alert"""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        if alert_type:
            alert.alert_type = AlertType(alert_type)
        if target_price is not None:
            alert.target_price = target_price
        if target_percent is not None:
            alert.target_percent = target_percent
        if message:
            alert.message = message
        if status:
            alert.alert_status = AlertStatus(status)
        
        db.commit()
        db.refresh(alert)
        
        return {
            "id": alert.id,
            "stock_symbol": alert.stock_symbol,
            "alert_type": alert.alert_type.value,
            "status": alert.alert_status.value,
            "message": "Alert updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """Delete an alert"""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        db.delete(alert)
        db.commit()
        
        return {"message": "Alert deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check")
async def check_alerts(
    user_id: int = Query(1, description="User ID"),
    db: Session = Depends(get_db)
):
    """Check and trigger alerts based on current prices"""
    try:
        # Get active alerts
        active_alerts = db.query(Alert).filter(
            Alert.user_id == user_id,
            Alert.alert_status == AlertStatus.ACTIVE
        ).all()
        
        data_collector = DataCollector()
        triggered_alerts = []
        
        for alert in active_alerts:
            try:
                # Get current stock data
                stock_data = await data_collector.get_stock_data_yahoo(
                    alert.stock_symbol, period="1d", interval="1m"
                )
                
                if stock_data is None or stock_data.empty:
                    continue
                
                current_price = stock_data['close'].iloc[-1]
                should_trigger = False
                
                # Check alert conditions
                if alert.alert_type == AlertType.PRICE_ABOVE and alert.target_price:
                    should_trigger = current_price >= alert.target_price
                elif alert.alert_type == AlertType.PRICE_BELOW and alert.target_price:
                    should_trigger = current_price <= alert.target_price
                elif alert.alert_type == AlertType.PERCENT_CHANGE and alert.target_percent:
                    # Calculate percent change from previous close
                    if len(stock_data) > 1:
                        prev_close = stock_data['close'].iloc[-2]
                        percent_change = ((current_price - prev_close) / prev_close) * 100
                        should_trigger = abs(percent_change) >= alert.target_percent
                
                if should_trigger:
                    alert.alert_status = AlertStatus.TRIGGERED
                    alert.triggered_at = datetime.utcnow()
                    triggered_alerts.append({
                        "id": alert.id,
                        "stock_symbol": alert.stock_symbol,
                        "current_price": current_price,
                        "message": alert.message
                    })
                
            except Exception as e:
                logger.error(f"Error checking alert {alert.id}: {str(e)}")
                continue
        
        db.commit()
        
        return {
            "checked_alerts": len(active_alerts),
            "triggered_alerts": triggered_alerts,
            "message": f"Checked {len(active_alerts)} alerts, triggered {len(triggered_alerts)}"
        }
        
    except Exception as e:
        logger.error(f"Error checking alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 