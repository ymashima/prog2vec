public class Test {
  /**
   * <!-- begin-user-doc --> <!-- end-user-doc -->
   * @generated
   */
  public void setId(String newId) {
    String oldId = id;
    id = newId;
    boolean oldIdESet = idESet;
    idESet = true;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(
          this, Notification.SET, PomPackage.PLUGIN_EXECUTION__ID, oldId, id, !oldIdESet));
  }
}